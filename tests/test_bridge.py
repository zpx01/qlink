"""The chat session's own logic (no model): parsing Qwen's call, the one-context loop in both
modes."""

from qgpu.bridge import Session, parse_call

CALL = """<tool_call>
<function=cuda>
<parameter=source>
__global__ void f(const int *a, int *y, int n) { int i = threadIdx.x; if (i < n) y[i] = 2 * a[i]; }
</parameter>
<parameter=args>
{"a": [1, 2], "n": 2}
</parameter>
</function>
</tool_call>"""

def test_parse_call():
    name, params = parse_call("Let me compute.\n\n" + CALL)
    assert name == "cuda" and params["args"] == '{"a": [1, 2], "n": 2}'
    assert params["source"].startswith("__global__ void f(const int *a")


class FakeTok:
    def apply_chat_template(self, messages, **_):
        return "|".join(m["content"] for m in messages)


def stub(turns, result, gpu=True, mode="cuda"):
    """A Session without a model: each assistant turn says the next of `turns`, every cuda
    call returns `result`. Records what is pushed into which context."""
    s = Session.__new__(Session)
    s.call_end, s.im_end, s.max_calls, s.cancel, s.think = 1, 2, 4, False, False
    s.mode = mode
    s.tok = FakeTok()
    s.reset()
    s.pushed, s.tools = [], []
    said = iter(turns)

    def assistant():
        return next(said), True
        yield

    def tool(call):
        s.tools.append(call)
        return result
        yield

    def enc(text):
        s.pushed.append((s.ctx is s.chat, text))
        return [0]

    s._assistant, s._cuda, s._calc, s._enc = assistant, tool, tool, enc
    s._needs_gpu = lambda user: gpu
    return s


def kinds(events):
    return [e.kind for e in events]


def test_no_calculation():
    s = stub(["Paris."], None, gpu=False)
    assert kinds(s.send("capital of France?")) == ["done"]
    assert not s.tools


def test_kernel_call_then_answer_in_one_context():
    """Advanced mode: Qwen writes the kernel in the chat itself (no separate passes); the
    router starts its turn inside the cuda call, and the answer sees the GPU's response."""
    rest = CALL.split("<parameter=source>\n", 1)[1]  # what Qwen writes after the forced start
    s = stub([rest, "y is [2, 4]."], {"y": [2, 4]})
    assert kinds(s.send("double 1 and 2")) == ["text", "done"]  # the forced start, shown
    assert len(s.tools) == 1 and "<function=cuda>" in s.tools[0]
    assert all(in_chat for in_chat, _ in s.pushed)  # one context: the chat
    pushed = "".join(t for _, t in s.pushed)
    assert "<parameter=source>\n" in pushed and '{"y": [2, 4]}' in pushed


def test_repeated_refused_call_is_not_run():
    s = stub([CALL, CALL, "Sorry."], {"error": "refused"}, gpu=False)
    events = list(s.send("hi"))
    assert any(e.kind == "refused" and e.data.get("repeat") for e in events)
    assert len(s.tools) == 1


def test_calls_are_capped():
    calls = [CALL.replace("2 *", f"{i} *") for i in range(10)]  # all different, all refused
    s = stub([*calls[:5], "Sorry."], {"error": "refused"}, gpu=False)
    list(s.send("hi"))
    assert len(s.tools) == 4  # max_calls


CALC_CALL = """<tool_call>
<function=calc>
<parameter=expression>
4*30 + 3*55
</parameter>
</function>
</tool_call>"""


def test_calc_mode_answer_without_a_call():
    s = stub(["Paris."], None, gpu=False, mode="calc")
    assert kinds(s.send("capital of France?")) == ["done"]
    assert not s.tools


def test_calc_mode_call_then_answer():
    s = stub(["Let me compute.\n\n" + CALC_CALL, "You spent 285."], {"result": 285}, mode="calc")
    # routed: the forced call's start shows as Qwen's text (the stub's turns stream none)
    assert kinds(s.send("apples and melons?")) == ["text", "done"]
    assert len(s.tools) == 1 and "<function=calc>" in s.tools[0]
    assert '{"result": 285}' in "".join(t for _, t in s.pushed)  # the response, in context


def test_calc_mode_routed_message_starts_with_the_call():
    """The router says a calculation: Qwen's turn starts inside the calc call, so it writes
    only the expression and cannot answer without the GPU."""
    rest = "4*30 + 3*55\n</parameter>\n</function>\n</tool_call>"
    s = stub([rest, "You spent 285."], {"result": 285}, mode="calc")
    list(s.send("apples and melons?"))
    assert "<parameter=expression>\n" in s.pushed[1][1]  # pushed before Qwen's first token
    assert len(s.tools) == 1 and "<function=calc>" in s.tools[0] and "4*30" in s.tools[0]


def test_calc_mode_repeated_refused_call_is_not_run():
    s = stub([CALC_CALL, CALC_CALL, "Sorry."], {"error": "refused"}, gpu=False, mode="calc")
    events = list(s.send("hi"))
    assert any(e.kind == "refused" and e.data.get("repeat") for e in events)
    assert len(s.tools) == 1
