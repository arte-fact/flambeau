// Protocol-level test for the C1.1 SSE → tool-call accumulator. Mirrors
// the logic embedded in `crates/server/assets/index.html`, run in
// isolation against a synthetic stream so we can assert the resulting
// tool-call state without DOM concerns.

// Reproduced verbatim from index.html (`makeSseParser`):
function makeSseParser(onFrame) {
  let buffer = "";
  return function feed(chunk) {
    buffer += chunk;
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = frame.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        onFrame(payload);
      }
    }
  };
}

// Mirrors the in-page tool-call accumulator state.
const toolCalls = new Map();
let assistantText = "";
let finishReason = null;

function onFrame(payload) {
  if (payload === "[DONE]") return;
  const obj = JSON.parse(payload);
  if (obj?.error) throw new Error(obj.error.message);
  const delta = obj?.choices?.[0]?.delta;
  const contentDelta = delta?.content;
  if (typeof contentDelta === "string" && contentDelta.length) {
    assistantText += contentDelta;
  }
  const tcDeltas = delta?.tool_calls;
  if (Array.isArray(tcDeltas)) {
    for (const tc of tcDeltas) {
      if (typeof tc?.index !== "number") continue;
      const idx = tc.index;
      let entry = toolCalls.get(idx);
      if (!entry) {
        entry = { id: "", name: "", args: "" };
        toolCalls.set(idx, entry);
      }
      if (typeof tc.id === "string" && tc.id) entry.id = tc.id;
      const fname = tc.function?.name;
      if (typeof fname === "string" && fname) entry.name = fname;
      const fargs = tc.function?.arguments;
      if (typeof fargs === "string" && fargs.length) entry.args += fargs;
    }
  }
  const fr = obj?.choices?.[0]?.finish_reason;
  if (fr) finishReason = fr;
}

const parser = makeSseParser(onFrame);

function frame(o) { return "data: " + JSON.stringify(o) + "\n\n"; }
const stream =
    frame({choices:[{delta:{role:"assistant"}}]})
  + frame({choices:[{delta:{content:"I'll check both cities.\n"}}]})
  + frame({choices:[{delta:{tool_calls:[{index:0,id:"call_a",type:"function",function:{name:"get_weather"}}]}}]})
  + frame({choices:[{delta:{tool_calls:[{index:1,id:"call_b",type:"function",function:{name:"get_weather"}}]}}]})
  + frame({choices:[{delta:{tool_calls:[{index:0,function:{arguments:"{\"city\":"}}]}}]})
  + frame({choices:[{delta:{tool_calls:[{index:1,function:{arguments:"{\"city\":"}}]}}]})
  + frame({choices:[{delta:{tool_calls:[{index:0,function:{arguments:"\"SF\"}"}}]}}]})
  + frame({choices:[{delta:{tool_calls:[{index:1,function:{arguments:"\"Tokyo\"}"}}]}}]})
  + frame({choices:[{delta:{},finish_reason:"tool_calls"}]})
  + "data: [DONE]\n\n";

// Feed in arbitrarily-split chunks (simulates real network buffering).
const splits = [37, 71, 119, 200, 273, 350, 412, 9999];
let cursor = 0;
for (const end of splits) {
  parser(stream.slice(cursor, end));
  cursor = end;
}

const fail = (msg) => { console.error("FAIL:", msg); process.exit(1); };

if (assistantText !== "I'll check both cities.\n")
  fail(`text mismatch: ${JSON.stringify(assistantText)}`);
if (finishReason !== "tool_calls") fail(`finish_reason=${finishReason}`);
if (toolCalls.size !== 2) fail(`expected 2 tool calls, got ${toolCalls.size}`);

const e0 = toolCalls.get(0), e1 = toolCalls.get(1);
if (e0.id !== "call_a" || e0.name !== "get_weather" || e0.args !== '{"city":"SF"}')
  fail(`tool_call[0]=${JSON.stringify(e0)}`);
if (e1.id !== "call_b" || e1.name !== "get_weather" || e1.args !== '{"city":"Tokyo"}')
  fail(`tool_call[1]=${JSON.stringify(e1)}`);

// Argument JSON must round-trip parse — proves the streamed args concatenated
// without losing or duplicating bytes across the chunk boundaries.
const a0 = JSON.parse(e0.args);
const a1 = JSON.parse(e1.args);
if (a0.city !== "SF" || a1.city !== "Tokyo") fail("args parse wrong");

console.log(`OK — text=${JSON.stringify(assistantText)} finish=${finishReason}`);
console.log(`     tool_calls[0]=${JSON.stringify(e0)}`);
console.log(`     tool_calls[1]=${JSON.stringify(e1)}`);
