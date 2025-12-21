"""
LM Studio API Client
====================

Client REST API pour LM Studio avec support des métriques de performance:
- TTFT (Time To First Token)
- Tokens per second (output/prompt)
- Streaming support
- Structured output (JSON mode)
"""

import time
import json
from dataclasses import dataclass, field
from typing import Optional, Generator, Any
from pathlib import Path

import httpx
import yaml
from openai import OpenAI


@dataclass
class CompletionMetrics:
    """Métriques collectées lors d'une complétion."""
    
    # Timing
    ttft_ms: float = 0.0                    # Time to first token (ms)
    total_time_ms: float = 0.0              # Temps total de génération
    
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Throughput
    output_tokens_per_sec: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    
    # Response
    finish_reason: str = ""
    model: str = ""
    
    # Raw response stats (from LM Studio)
    raw_stats: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convertit les métriques en dictionnaire."""
        return {
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "prompt_tokens_per_sec": self.prompt_tokens_per_sec,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "raw_stats": self.raw_stats,
        }


@dataclass
class CompletionResult:
    """Résultat d'une complétion avec métriques."""
    
    content: str
    metrics: CompletionMetrics
    success: bool = True
    error: Optional[str] = None
    
    # Pour structured output
    json_valid: bool = False
    parsed_json: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convertit le résultat en dictionnaire."""
        return {
            "content": self.content,
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "error": self.error,
            "json_valid": self.json_valid,
            "parsed_json": self.parsed_json,
        }


class LMStudioClient:
    """
    Client pour l'API LM Studio.
    
    Compatible avec l'API OpenAI pour les chat completions.
    Collecte automatiquement les métriques de performance.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 120.0,
        config_path: Optional[Path] = None,
    ):
        """
        Initialise le client LM Studio.
        
        Args:
            base_url: URL de base de l'API LM Studio
            timeout: Timeout en secondes pour les requêtes
            config_path: Chemin vers le fichier de configuration des modèles
        """
        self.base_url = base_url
        self.timeout = timeout
        
        # Client OpenAI-compatible
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LM Studio n'exige pas de clé
            timeout=timeout,
        )
        
        # Client HTTP pour les requêtes directes
        self.http_client = httpx.Client(
            base_url=base_url.replace("/v1", ""),
            timeout=timeout,
        )
        
        # Charger la configuration des modèles si fournie
        self.models_config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.models_config = yaml.safe_load(f)
    
    def list_models(self) -> list[dict]:
        """Liste les modèles disponibles sur LM Studio."""
        try:
            response = self.client.models.list()
            return [{"id": m.id, "object": m.object} for m in response.data]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_model_info(self, model_id: str) -> dict:
        """Récupère les informations d'un modèle."""
        try:
            response = self.http_client.get(f"/v1/models/{model_id}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def load_model(self, model_id: str) -> bool:
        """
        Charge un modèle dans LM Studio.
        
        Note: Nécessite que LM Studio supporte le chargement via API.
        """
        try:
            # LM Studio charge automatiquement le modèle à la première requête
            # On fait une requête minimale pour forcer le chargement
            self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
    
    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 512,
        stream: bool = True,
        response_format: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        Effectue une complétion avec collecte de métriques.
        
        Args:
            model: ID du modèle LM Studio
            messages: Liste de messages (format OpenAI)
            temperature: Température d'échantillonnage
            top_p: Top-p sampling
            max_tokens: Nombre maximum de tokens en sortie
            stream: Utiliser le streaming (recommandé pour TTFT)
            response_format: Format de réponse (ex: {"type": "json_object"})
            stop: Séquences d'arrêt
            
        Returns:
            CompletionResult avec contenu et métriques
        """
        metrics = CompletionMetrics(model=model)
        
        try:
            start_time = time.perf_counter()
            first_token_time = None
            content_chunks = []
            
            if stream:
                # Mode streaming pour mesurer TTFT précisément
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True,
                    response_format=response_format,
                    stop=stop,
                    **kwargs,
                )
                
                for chunk in response:
                    if first_token_time is None and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            first_token_time = time.perf_counter()
                            metrics.ttft_ms = (first_token_time - start_time) * 1000
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunks.append(chunk.choices[0].delta.content)
                    
                    # Récupérer finish_reason du dernier chunk
                    if chunk.choices and chunk.choices[0].finish_reason:
                        metrics.finish_reason = chunk.choices[0].finish_reason
                
                content = "".join(content_chunks)
                
            else:
                # Mode non-streaming
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                    response_format=response_format,
                    stop=stop,
                    **kwargs,
                )
                
                first_token_time = time.perf_counter()
                metrics.ttft_ms = (first_token_time - start_time) * 1000
                
                content = response.choices[0].message.content or ""
                metrics.finish_reason = response.choices[0].finish_reason or ""
                
                # Métriques d'usage
                if response.usage:
                    metrics.prompt_tokens = response.usage.prompt_tokens
                    metrics.completion_tokens = response.usage.completion_tokens
                    metrics.total_tokens = response.usage.total_tokens
            
            end_time = time.perf_counter()
            metrics.total_time_ms = (end_time - start_time) * 1000
            
            # Calculer le throughput
            if metrics.total_time_ms > 0:
                # Estimer les tokens de complétion si pas fournis
                if metrics.completion_tokens == 0:
                    # Estimation grossière: ~4 caractères par token
                    metrics.completion_tokens = len(content) // 4
                
                generation_time_sec = (end_time - (first_token_time or start_time))
                if generation_time_sec > 0:
                    metrics.output_tokens_per_sec = metrics.completion_tokens / generation_time_sec
                
                if first_token_time and metrics.prompt_tokens > 0:
                    prompt_time_sec = first_token_time - start_time
                    if prompt_time_sec > 0:
                        metrics.prompt_tokens_per_sec = metrics.prompt_tokens / prompt_time_sec
            
            # Vérifier si JSON valide (pour structured output)
            json_valid = False
            parsed_json = None
            if response_format and response_format.get("type") == "json_object":
                try:
                    parsed_json = json.loads(content)
                    json_valid = True
                except json.JSONDecodeError:
                    json_valid = False
            
            return CompletionResult(
                content=content,
                metrics=metrics,
                success=True,
                json_valid=json_valid,
                parsed_json=parsed_json,
            )
            
        except Exception as e:
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
            return CompletionResult(
                content="",
                metrics=metrics,
                success=False,
                error=str(e),
            )
    
    def complete_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_retries: int = 3,
        **kwargs,
    ) -> CompletionResult:
        """
        Complétion avec retry automatique en cas d'erreur.
        """
        last_error = None
        for attempt in range(max_retries):
            result = self.complete(model=model, messages=messages, **kwargs)
            if result.success:
                return result
            last_error = result.error
            time.sleep(1 * (attempt + 1))  # Backoff exponentiel simple
        
        return CompletionResult(
            content="",
            metrics=CompletionMetrics(model=model),
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
        )
    
    def warmup(self, model: str, num_requests: int = 3) -> list[CompletionMetrics]:
        """
        Effectue des requêtes de warm-up (non comptées dans les benchmarks).
        
        Args:
            model: ID du modèle
            num_requests: Nombre de requêtes de warm-up
            
        Returns:
            Liste des métriques des requêtes de warm-up
        """
        warmup_metrics = []
        warmup_message = [{"role": "user", "content": "Hello, this is a warmup request."}]
        
        for _ in range(num_requests):
            result = self.complete(
                model=model,
                messages=warmup_message,
                max_tokens=20,
                stream=True,
            )
            warmup_metrics.append(result.metrics)
            time.sleep(0.5)
        
        return warmup_metrics
    
    def batch_complete(
        self,
        model: str,
        prompts: list[list[dict]],
        cooldown: float = 1.0,
        **kwargs,
    ) -> Generator[CompletionResult, None, None]:
        """
        Effectue plusieurs complétions avec cooldown entre chaque.
        
        Args:
            model: ID du modèle
            prompts: Liste de listes de messages
            cooldown: Temps de pause entre les requêtes (secondes)
            
        Yields:
            CompletionResult pour chaque prompt
        """
        for i, messages in enumerate(prompts):
            result = self.complete(model=model, messages=messages, **kwargs)
            yield result
            
            if i < len(prompts) - 1:
                time.sleep(cooldown)
    
    def health_check(self) -> dict:
        """Vérifie la santé du serveur LM Studio."""
        try:
            response = self.http_client.get("/v1/models")
            return {
                "status": "healthy",
                "models_available": response.status_code == 200,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def close(self):
        """Ferme les connexions HTTP."""
        self.http_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Fonctions utilitaires

def format_messages(
    user_content: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """
    Formate les messages pour l'API chat.
    
    Args:
        user_content: Contenu du message utilisateur
        system_prompt: Prompt système optionnel
        
    Returns:
        Liste de messages formatés
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def create_classification_prompt(
    text: str,
    labels: list[str],
    instruction: Optional[str] = None,
) -> str:
    """
    Crée un prompt pour une tâche de classification.
    
    Args:
        text: Texte à classifier
        labels: Labels possibles
        instruction: Instruction personnalisée
        
    Returns:
        Prompt formaté
    """
    if instruction is None:
        instruction = "Classify the following text."
    
    labels_str = ", ".join(labels)
    return f"{instruction}\n\nText: {text}\n\nPossible labels: {labels_str}\n\nLabel:"

