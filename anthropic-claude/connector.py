"""
Copyright start
MIT License
Copyright (c) 2026 Fortinet Inc
Copyright end
"""

from .operations import operations, check_health
from connectors.core.connector import Connector, get_logger, ConnectorError

logger = get_logger('anthropic-claude')


class AnthropicClaude(Connector):
    def execute(self, config, operation, params, *args, **kwargs):
        try:
            action = operations.get(operation)
            if not action:
                raise ConnectorError("Unsupported operation: {}".format(operation))
            logger.info('Executing action: {}'.format(operation))
            return action(config, params)
        except Exception as err:
            logger.exception("An exception occurred [{}]".format(err))
            raise ConnectorError("An exception occurred [{}]".format(err))

    def check_health(self, config):
        logger.info('Starting health check for Anthropic Claude connector')
        check_health(config)
        logger.info('Health check completed successfully')
