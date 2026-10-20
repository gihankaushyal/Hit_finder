/**
 * A reader for the top level of a pickle, used to find the `epoch` stored in a torch checkpoint without
 * loading the model. It is a data parser, not an unpickler: GLOBAL, REDUCE, NEWOBJ and similar opcodes
 * only produce an opaque placeholder, so nothing in the file is ever imported, called or constructed.
 * Anything unexpected (an unknown opcode, a truncated stream, a count or size over a limit) gives null.
 */

/** Stands for anything that is not plain data: classes, tensors, persistent ids, bytes. */
export const OPAQUE: unique symbol = Symbol("opaque");

export type PickleValue = number | string | boolean | null | typeof OPAQUE | PickleValue[] | Map<string, PickleValue>;

export const MAX_OPCODES = 2_000_000;
export const MAX_MARKS = 10_000;
export const MAX_STACK = 200_000;
export const MAX_MEMO = 1_000_000;
export const MAX_STRING_BYTES = 1_000_000;
const MAX_GLOBAL_LINE = 256;
const MAX_LONG_BYTES = 64;

class Bail extends Error {}
const bail = (): never => {
  throw new Bail();
};

/** The top-level dict of the pickle in `buf`, or null if it is not a plain dict or cannot be read safely. */
export function readPickleTopLevel(buf: Uint8Array): Map<string, PickleValue> | null {
  try {
    const result = run(buf);
    return result instanceof Map ? result : null;
  } catch {
    return null;
  }
}

function run(buf: Uint8Array): PickleValue {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const stack: PickleValue[] = [];
  const marks: number[] = [];
  const memo = new Map<number, PickleValue>();
  let pos = 0;
  let ops = 0;

  const need = (n: number) => {
    if (n < 0 || pos + n > buf.length) bail();
  };
  const u8 = () => (need(1), buf[pos++]);
  const u16 = () => (need(2), ((pos += 2), view.getUint16(pos - 2, true)));
  const u32 = () => (need(4), ((pos += 4), view.getUint32(pos - 4, true)));
  const i32 = () => (need(4), ((pos += 4), view.getInt32(pos - 4, true)));
  const push = (v: PickleValue) => {
    if (stack.length >= MAX_STACK) bail();
    stack.push(v);
  };
  const pop = (): PickleValue => (stack.length ? (stack.pop() as PickleValue) : bail());
  const top = (): PickleValue => (stack.length ? stack[stack.length - 1] : bail());
  const popMark = (): PickleValue[] => {
    const at = marks.pop();
    if (at === undefined || at > stack.length) bail();
    return stack.splice(at as number);
  };
  const text = (n: number): string => {
    if (n > MAX_STRING_BYTES) bail();
    need(n);
    const s = new TextDecoder("utf-8", { fatal: false }).decode(buf.subarray(pos, pos + n));
    pos += n;
    return s;
  };
  const skip = (n: number) => {
    need(n);
    pos += n;
  };
  const line = (): void => {
    const start = pos;
    while (pos < buf.length && buf[pos] !== 0x0a) {
      if (pos - start > MAX_GLOBAL_LINE) bail();
      pos++;
    }
    need(1);
    pos++; // the newline
  };
  const long = (n: number): number | null => {
    if (n > MAX_LONG_BYTES) bail();
    need(n);
    let value = 0n;
    for (let i = n - 1; i >= 0; i--) value = (value << 8n) | BigInt(buf[pos + i]);
    if (n > 0 && buf[pos + n - 1] & 0x80) value -= 1n << BigInt(8 * n);
    pos += n;
    const asNumber = Number(value);
    return Number.isSafeInteger(asNumber) ? asNumber : null;
  };
  const memoPut = (key: number) => {
    if (memo.size >= MAX_MEMO && !memo.has(key)) bail();
    memo.set(key, top());
  };
  const target = (): Map<string, PickleValue> | PickleValue[] | typeof OPAQUE => {
    const t = top();
    if (t instanceof Map || Array.isArray(t) || t === OPAQUE) return t;
    return bail();
  };
  const setItem = (t: ReturnType<typeof target>, key: PickleValue, value: PickleValue) => {
    if (t instanceof Map && typeof key === "string") t.set(key, value); // non-string keys are not needed
  };

  for (;;) {
    if (++ops > MAX_OPCODES) bail();
    const code = u8();
    switch (code) {
      case 0x80: skip(1); break; // PROTO
      case 0x95: skip(8); break; // FRAME
      case 0x2e: { // STOP
        return pop();
      }
      case 0x28: // MARK
        if (marks.length >= MAX_MARKS) bail();
        marks.push(stack.length);
        break;
      case 0x7d: push(new Map()); break; // EMPTY_DICT
      case 0x5d: push([]); break; // EMPTY_LIST
      case 0x29: push([]); break; // EMPTY_TUPLE
      case 0x8f: push(OPAQUE); break; // EMPTY_SET
      case 0x4e: push(null); break; // NONE
      case 0x88: push(true); break; // NEWTRUE
      case 0x89: push(false); break; // NEWFALSE
      case 0x4b: push(u8()); break; // BININT1
      case 0x4d: push(u16()); break; // BININT2
      case 0x4a: push(i32()); break; // BININT
      case 0x8a: push(long(u8())); break; // LONG1
      case 0x8b: push(long(u32())); break; // LONG4
      case 0x47: // BINFLOAT (big-endian)
        need(8);
        push(view.getFloat64(pos, false));
        pos += 8;
        break;
      case 0x58: push(text(u32())); break; // BINUNICODE
      case 0x8c: push(text(u8())); break; // SHORT_BINUNICODE
      case 0x8d: { // BINUNICODE8
        need(8);
        const n = Number(view.getBigUint64(pos, true));
        pos += 8;
        push(text(n));
        break;
      }
      case 0x54: push(text(u32())); break; // BINSTRING
      case 0x55: push(text(u8())); break; // SHORT_BINSTRING
      case 0x42: skip(u32()); push(OPAQUE); break; // BINBYTES
      case 0x43: skip(u8()); push(OPAQUE); break; // SHORT_BINBYTES
      case 0x96: { // BYTEARRAY8
        need(8);
        const n = Number(view.getBigUint64(pos, true));
        pos += 8;
        skip(n);
        push(OPAQUE);
        break;
      }
      case 0x63: line(); line(); push(OPAQUE); break; // GLOBAL
      case 0x93: pop(); pop(); push(OPAQUE); break; // STACK_GLOBAL
      case 0x52: pop(); pop(); push(OPAQUE); break; // REDUCE
      case 0x81: pop(); pop(); push(OPAQUE); break; // NEWOBJ
      case 0x92: pop(); pop(); pop(); push(OPAQUE); break; // NEWOBJ_EX
      case 0x62: pop(); break; // BUILD: the state is dropped, the object stays
      case 0x51: pop(); push(OPAQUE); break; // BINPERSID
      case 0x85: push([pop()]); break; // TUPLE1
      case 0x86: { const b = pop(); const a = pop(); push([a, b]); break; } // TUPLE2
      case 0x87: { const c = pop(); const b = pop(); const a = pop(); push([a, b, c]); break; } // TUPLE3
      case 0x74: push(popMark()); break; // TUPLE
      case 0x6c: push(popMark()); break; // LIST
      case 0x64: { // DICT
        const items = popMark();
        const m = new Map<string, PickleValue>();
        for (let i = 0; i + 1 < items.length; i += 2) if (typeof items[i] === "string") m.set(items[i] as string, items[i + 1]);
        push(m);
        break;
      }
      case 0x61: { const v = pop(); const t = target(); if (Array.isArray(t)) t.push(v); break; } // APPEND
      case 0x65: { // APPENDS
        const items = popMark();
        const t = target();
        if (Array.isArray(t)) for (const v of items) t.push(v);
        break;
      }
      case 0x73: { const v = pop(); const k = pop(); setItem(target(), k, v); break; } // SETITEM
      case 0x75: { // SETITEMS
        const items = popMark();
        const t = target();
        for (let i = 0; i + 1 < items.length; i += 2) setItem(t, items[i], items[i + 1]);
        break;
      }
      case 0x90: popMark(); break; // ADDITEMS
      case 0x91: popMark(); push(OPAQUE); break; // FROZENSET
      case 0x71: memoPut(u8()); break; // BINPUT
      case 0x72: memoPut(u32()); break; // LONG_BINPUT
      case 0x94: memoPut(memo.size); break; // MEMOIZE
      case 0x68: { const k = u8(); if (!memo.has(k)) bail(); push(memo.get(k) as PickleValue); break; } // BINGET
      case 0x6a: { const k = u32(); if (!memo.has(k)) bail(); push(memo.get(k) as PickleValue); break; } // LONG_BINGET
      case 0x30: pop(); break; // POP
      case 0x32: push(top()); break; // DUP
      case 0x31: popMark(); break; // POP_MARK
      default: bail();
    }
  }
}
