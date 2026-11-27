import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from v2.nacos import ClientConfig
from v2.nacos.ai.model.ai_request import QueryMcpServerRequest, \
	QueryPromptRequest
from v2.nacos.ai.model.ai_response import QueryMcpServerResponse, \
	QueryPromptResponse
from v2.nacos.ai.remote.ai_grpc_client_proxy import AIGRPCClientProxy
from v2.nacos.common.nacos_exception import NacosException, NOT_FOUND, \
	NOT_MODIFIED, SERVER_ERROR

RESPONSE_FAILED = 500


def make_proxy(error_code: int, message: str,
			   response_class=QueryPromptResponse) -> AIGRPCClientProxy:
	"""Build a proxy whose server always answers with the given error code."""
	proxy = AIGRPCClientProxy.__new__(AIGRPCClientProxy)
	proxy.rpc_client = SimpleNamespace(request=AsyncMock(
		return_value=response_class(resultCode=RESPONSE_FAILED,
									errorCode=error_code,
									message=message)))
	proxy.nacos_server_connector = SimpleNamespace(
		inject_security_info=AsyncMock())
	proxy.client_config = ClientConfig()
	proxy.app_name = "test-app"
	proxy.logger = logging.getLogger("test-ai-grpc-client-proxy")
	return proxy


class TestAIGRPCClientProxyLogging(unittest.IsolatedAsyncioTestCase):

	async def query_prompt(self, proxy: AIGRPCClientProxy) -> NacosException:
		request = QueryPromptRequest(namespaceId="public", promptKey="key")
		with self.assertRaises(NacosException) as ctx:
			await proxy.request_ai_server(request, QueryPromptResponse)
		return ctx.exception

	def assert_single_warning(self, logs, message: str):
		self.assertEqual([logging.WARNING],
						 [record.levelno for record in logs.records])
		self.assertIn(message, logs.output[0])

	async def test_not_modified_response_is_not_logged(self):
		proxy = make_proxy(NOT_MODIFIED, "prompt data is up to date")

		with self.assertNoLogs(proxy.logger):
			exception = await self.query_prompt(proxy)

		self.assertEqual(NOT_MODIFIED, exception.error_code)

	async def test_not_found_response_is_logged_as_warning(self):
		proxy = make_proxy(NOT_FOUND, "prompt not found")

		with self.assertLogs(proxy.logger, level=logging.WARNING) as logs:
			exception = await self.query_prompt(proxy)

		self.assertEqual(NOT_FOUND, exception.error_code)
		self.assert_single_warning(logs, "prompt not found")

	async def test_not_found_mcp_response_is_logged_as_warning(self):
		proxy = make_proxy(NOT_FOUND, "mcp server not found",
						   QueryMcpServerResponse)
		request = QueryMcpServerRequest(namespaceId="public", mcpName="mcp")

		with self.assertLogs(proxy.logger, level=logging.WARNING) as logs:
			with self.assertRaises(NacosException) as ctx:
				await proxy.request_ai_server(request, QueryMcpServerResponse)

		self.assertEqual(NOT_FOUND, ctx.exception.error_code)
		self.assert_single_warning(logs, "mcp server not found")

	async def test_server_error_response_is_still_logged_as_error(self):
		proxy = make_proxy(SERVER_ERROR, "internal server error")

		with self.assertLogs(proxy.logger, level=logging.ERROR) as logs:
			exception = await self.query_prompt(proxy)

		self.assertEqual(SERVER_ERROR, exception.error_code)
		self.assertIn("internal server error", logs.output[0])
