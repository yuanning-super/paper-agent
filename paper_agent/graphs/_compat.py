"""langgraph prebuilt agent 兼容层。

不同版本 API 不一致：
- langgraph >= 1.3：prebuilt.create_agent(model, tools, system_prompt=...)
- langgraph 1.0-1.2：prebuilt.create_react_agent(model, tools, prompt=...)
- langgraph 0.x：prebuilt.create_react_agent(model, tools, prompt=...)

统一对外暴露 system_prompt 参数的工厂。
"""

import inspect


def get_create_agent():
    try:
        from langgraph.prebuilt import create_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        create_agent = create_react_agent

    params = inspect.signature(create_agent).parameters
    if "system_prompt" in params:
        return create_agent

    def _factory(**kwargs):
        if "system_prompt" in kwargs:
            kwargs["prompt"] = kwargs.pop("system_prompt")
        return create_agent(**kwargs)

    return _factory
