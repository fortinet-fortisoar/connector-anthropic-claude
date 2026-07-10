"""
Copyright start
MIT License
Copyright (c) 2026 Fortinet Inc
Copyright end
"""

import json
import requests
from connectors.core.connector import get_logger, ConnectorError
from .constants import ANTHROPIC_API_BASE, DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS

logger = get_logger('anthropic-claude')


def _get_headers(config):
    api_key = config.get('api_key', '').strip()
    if not api_key:
        raise ConnectorError("API Key is required")
    anthropic_version = config.get('anthropic_version', '').strip() or DEFAULT_ANTHROPIC_VERSION
    return {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }


def _get_model(config, params):
    return (params.get('model') or config.get('model') or 'claude-sonnet-4-6').strip()


def _make_request(config, method, endpoint, payload=None):
    headers = _get_headers(config)
    verify_ssl = config.get('verify_ssl', True)
    url = "{}/{}".format(ANTHROPIC_API_BASE.rstrip('/'), endpoint.lstrip('/'))

    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, verify=verify_ssl, timeout=60)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload, verify=verify_ssl, timeout=120)
        else:
            raise ConnectorError("Unsupported HTTP method: {}".format(method))

        if response.ok:
            return response.json()

        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', response.text)
        except Exception:
            error_msg = response.text

        raise ConnectorError("Claude API Error [HTTP {}]: {}".format(response.status_code, error_msg))

    except ConnectorError:
        raise
    except requests.exceptions.SSLError as e:
        raise ConnectorError(
            "SSL verification failed. Disable 'Verify SSL' in connector config if using self-signed cert. Details: {}".format(
                str(e)))
    except requests.exceptions.ConnectionError as e:
        raise ConnectorError("Connection error: {}".format(str(e)))
    except requests.exceptions.Timeout:
        raise ConnectorError("Request timed out connecting to Anthropic API")
    except Exception as e:
        logger.exception("Unexpected error during API request")
        raise ConnectorError("Unexpected error: {}".format(str(e)))


def _parse_conversation_history(conversation_history):
    if not conversation_history:
        return []
    if isinstance(conversation_history, list):
        return conversation_history
    if isinstance(conversation_history, str):
        try:
            parsed = json.loads(conversation_history)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _extract_text_from_content(content_blocks):
    if not content_blocks:
        return ''
    return ''.join(
        block.get('text', '')
        for block in content_blocks
        if block.get('type') == 'text'
    )


def send_message(config, params):
    """Send a text prompt to Claude and receive a response."""
    model = _get_model(config, params)
    prompt = params.get('prompt', '').strip()
    if not prompt:
        raise ConnectorError("Prompt is required")

    system_prompt = params.get('system_prompt', '').strip()
    max_tokens = params.get('max_tokens') or DEFAULT_MAX_TOKENS
    temperature = params.get('temperature')
    stop_sequences_raw = params.get('stop_sequences', '')
    conversation_history = params.get('conversation_history')

    messages = _parse_conversation_history(conversation_history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": messages,
    }

    if system_prompt:
        payload["system"] = system_prompt

    if temperature is not None and str(temperature).strip():
        try:
            temp_val = float(temperature)
            if not 0.0 <= temp_val <= 1.0:
                raise ConnectorError("Temperature must be between 0.0 and 1.0")
            payload["temperature"] = temp_val
        except ValueError:
            raise ConnectorError("Temperature must be a number between 0.0 and 1.0")

    if stop_sequences_raw:
        if isinstance(stop_sequences_raw, list):
            payload["stop_sequences"] = stop_sequences_raw
        elif isinstance(stop_sequences_raw, str):
            sequences = [s.strip() for s in stop_sequences_raw.split(',') if s.strip()]
            if sequences:
                payload["stop_sequences"] = sequences

    result = _make_request(config, 'POST', '/v1/messages', payload)

    content_blocks = result.get('content', [])
    output_text = _extract_text_from_content(content_blocks)
    usage = result.get('usage', {})

    return {
        "id": result.get("id"),
        "model": result.get("model"),
        "role": result.get("role"),
        "content": output_text,
        "content_blocks": content_blocks,
        "stop_reason": result.get("stop_reason"),
        "stop_sequence": result.get("stop_sequence"),
        "usage": usage,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def chat_with_image(config, params):
    """Send an image with a text prompt to Claude for visual analysis."""
    model = _get_model(config, params)
    prompt = params.get('prompt', '').strip()
    if not prompt:
        raise ConnectorError("Prompt is required")

    image_url = params.get('image_url', '').strip()
    image_base64 = params.get('image_base64', '').strip()
    media_type = params.get('media_type', 'image/jpeg')
    system_prompt = params.get('system_prompt', '').strip()
    max_tokens = params.get('max_tokens') or DEFAULT_MAX_TOKENS
    temperature = params.get('temperature')

    if not image_url and not image_base64:
        raise ConnectorError("Either 'Image URL' or 'Base64 Image Data' must be provided")
    if image_url and image_base64:
        raise ConnectorError("Provide either 'Image URL' or 'Base64 Image Data', not both")

    content = []

    if image_url:
        content.append({
            "type": "image",
            "source": {"type": "url", "url": image_url},
        })
    else:
        # Strip data URI prefix if present (e.g. "data:image/png;base64,")
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64,
            },
        })

    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": content}],
    }

    if system_prompt:
        payload["system"] = system_prompt

    if temperature is not None and str(temperature).strip():
        try:
            temp_val = float(temperature)
            if not 0.0 <= temp_val <= 1.0:
                raise ConnectorError("Temperature must be between 0.0 and 1.0")
            payload["temperature"] = temp_val
        except ValueError:
            raise ConnectorError("Temperature must be a number between 0.0 and 1.0")

    result = _make_request(config, 'POST', '/v1/messages', payload)

    content_blocks = result.get('content', [])
    output_text = _extract_text_from_content(content_blocks)
    usage = result.get('usage', {})

    return {
        "id": result.get("id"),
        "model": result.get("model"),
        "role": result.get("role"),
        "content": output_text,
        "content_blocks": content_blocks,
        "stop_reason": result.get("stop_reason"),
        "usage": usage,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def count_tokens(config, params):
    """Count the tokens in a message without generating a response."""
    model = _get_model(config, params)
    prompt = params.get('prompt', '').strip()
    if not prompt:
        raise ConnectorError("Prompt is required")

    system_prompt = params.get('system_prompt', '').strip()
    conversation_history = params.get('conversation_history')

    messages = _parse_conversation_history(conversation_history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
    }

    if system_prompt:
        payload["system"] = system_prompt

    result = _make_request(config, 'POST', '/v1/messages/count_tokens', payload)

    return {
        "input_tokens": result.get("input_tokens"),
        "model": model,
    }


def list_models(config, params):
    """Retrieve all available Claude models from the Anthropic API."""
    result = _make_request(config, 'GET', '/v1/models')
    return result


def check_health(config):
    try:
        result = list_models(config, {})
        if result and result.get('data') is not None:
            return True
        raise ConnectorError("Unexpected response from Claude API during health check")
    except ConnectorError:
        raise
    except Exception as err:
        logger.exception("Health check failed")
        raise ConnectorError("Health check failed: {}".format(str(err)))


operations = {
    'send_message': send_message,
    'chat_with_image': chat_with_image,
    'count_tokens': count_tokens,
    'list_models': list_models,
}
