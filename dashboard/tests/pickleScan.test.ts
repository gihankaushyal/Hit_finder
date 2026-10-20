import { describe, expect, it } from "vitest";
import { OPAQUE, readPickleTopLevel } from "../server/lib/pickleScan";

// A tiny protocol-2 encoder, enough to build the shapes torch.save writes.
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const op = (...bytes: number[]) => Buffer.from(bytes);
const str = (s: string) => Buffer.concat([op(0x58), u32(Buffer.byteLength(s)), Buffer.from(s)]); // BINUNICODE
const int = (n: number) => {
  if (n >= 0 && n < 256) return op(0x4b, n); // BININT1
  if (n >= 0 && n < 65536) return op(0x4d, n & 255, n >> 8); // BININT2
  const b = Buffer.alloc(5); b[0] = 0x4a; b.writeInt32LE(n, 1); return b; // BININT
};
const float = (x: number) => { const b = Buffer.alloc(9); b[0] = 0x47; b.writeDoubleBE(x, 1); return b; }; // BINFLOAT
const long1 = (x: bigint) => { // LONG1: little-endian two's complement
  let hex = BigInt.asUintN(72, x).toString(16).padStart(18, "0");
  const bytes = Buffer.from(hex, "hex").reverse();
  return Buffer.concat([op(0x8a, bytes.length), bytes]);
};
const NONE = op(0x4e), TRUE = op(0x88), FALSE = op(0x89), STOP = op(0x2e), PROTO2 = op(0x80, 2);
const MARK = op(0x28), EMPTY_DICT = op(0x7d), SETITEMS = op(0x75), EMPTY_LIST = op(0x5d), APPENDS = op(0x65);
const put = (n: number) => op(0x71, n); // BINPUT
const get = (n: number) => op(0x68, n); // BINGET
const dict = (...kv: Buffer[]) => Buffer.concat([EMPTY_DICT, MARK, ...kv, SETITEMS]);
const pickle = (...parts: Buffer[]) => Buffer.concat([PROTO2, ...parts, STOP]);

describe("readPickleTopLevel", () => {
  it("reads strings, numbers, booleans and None at the top level of a dict", () => {
    const m = readPickleTopLevel(pickle(dict(str("epoch"), int(187), str("val_f1"), float(0.93), str("ok"), TRUE, str("no"), FALSE, str("n"), NONE, str("backbone"), str("resnet18"))));
    expect(m).not.toBeNull();
    expect(m!.get("epoch")).toBe(187);
    expect(m!.get("val_f1")).toBeCloseTo(0.93);
    expect(m!.get("ok")).toBe(true);
    expect(m!.get("no")).toBe(false);
    expect(m!.get("n")).toBeNull();
    expect(m!.get("backbone")).toBe("resnet18");
  });

  it.each([[0], [255], [256], [65535], [65536], [-1], [123456789]])("reads the integer %i", (n) => {
    expect(readPickleTopLevel(pickle(dict(str("epoch"), int(n))))!.get("epoch")).toBe(n);
  });

  it("reads a LONG1 integer that fits in a double and refuses one that does not", () => {
    expect(readPickleTopLevel(pickle(dict(str("epoch"), long1(2n ** 40n))))!.get("epoch")).toBe(2 ** 40);
    expect(readPickleTopLevel(pickle(dict(str("epoch"), long1(2n ** 62n))))!.get("epoch")).toBeNull();
  });

  it("only returns top-level keys: a nested dict's 'epoch' is not the checkpoint's epoch", () => {
    const m = readPickleTopLevel(pickle(dict(str("optimizer"), dict(str("epoch"), int(1)), str("epoch"), int(2))));
    expect(m!.get("epoch")).toBe(2);
    expect(m!.get("optimizer")).toBeInstanceOf(Map);
  });

  it("an object built by GLOBAL and REDUCE is opaque and nothing is run", () => {
    // collections.OrderedDict() and os.system('echo hi'): both only become placeholders.
    const global = (mod: string, name: string) => Buffer.concat([op(0x63), Buffer.from(`${mod}\n${name}\n`)]);
    const reduce = (g: Buffer, arg: Buffer) => Buffer.concat([g, arg, op(0x85), op(0x52)]); // arg TUPLE1 REDUCE
    const m = readPickleTopLevel(pickle(dict(str("model_state_dict"), reduce(global("collections", "OrderedDict"), op(0x29)), str("evil"), reduce(global("os", "system"), str("echo hi")), str("epoch"), int(3))));
    expect(m!.get("model_state_dict")).toBe(OPAQUE);
    expect(m!.get("evil")).toBe(OPAQUE);
    expect(m!.get("epoch")).toBe(3);
  });

  it("follows memo references", () => {
    const m = readPickleTopLevel(pickle(dict(str("a"), Buffer.concat([str("shared"), put(5)]), str("b"), get(5))));
    expect(m!.get("a")).toBe("shared");
    expect(m!.get("b")).toBe("shared");
  });

  it("lists are read, and a persistent id (a tensor storage) is opaque", () => {
    const m = readPickleTopLevel(pickle(dict(str("l"), Buffer.concat([EMPTY_LIST, MARK, int(1), int(2), APPENDS]), str("t"), Buffer.concat([str("storage"), op(0x51)]))));
    expect(m!.get("l")).toEqual([1, 2]);
    expect(m!.get("t")).toBe(OPAQUE);
  });

  it("a key named __proto__ is an ordinary entry and pollutes nothing", () => {
    const m = readPickleTopLevel(pickle(dict(str("__proto__"), int(1), str("epoch"), int(2))));
    expect(m!.get("__proto__")).toBe(1);
    expect(({} as Record<string, unknown>).epoch).toBeUndefined();
    expect(Object.getPrototypeOf({})).toBe(Object.prototype);
  });

  it.each([
    ["an unknown opcode", pickle(dict(str("epoch"), op(0xff)))],
    ["a truncated stream", pickle(dict(str("epoch"), int(1))).subarray(0, 12)],
    ["a missing STOP", Buffer.concat([PROTO2, dict(str("epoch"), int(1))])],
    ["a string longer than the data", Buffer.concat([PROTO2, EMPTY_DICT, MARK, str("k"), op(0x58), u32(0x7fffffff), Buffer.from("x"), SETITEMS, STOP])],
    ["a top-level list", pickle(Buffer.concat([EMPTY_LIST, MARK, int(1), APPENDS]))],
    ["an empty buffer", Buffer.alloc(0)],
    ["a reference to an unset memo slot", pickle(dict(str("a"), get(9)))],
  ])("returns null for %s", (_name, buf) => {
    expect(readPickleTopLevel(buf)).toBeNull();
  });

  it("refuses more open MARKs than any real pickle has, even when the rest of the stream is a valid dict", () => {
    // Without the limit this would parse: the marks are simply never closed, then STOP returns the dict.
    const many = Buffer.concat([PROTO2, ...Array.from({ length: 20_000 }, () => MARK), EMPTY_DICT, STOP]);
    expect(readPickleTopLevel(many)).toBeNull();
    const few = Buffer.concat([PROTO2, ...Array.from({ length: 50 }, () => MARK), EMPTY_DICT, STOP]);
    expect(readPickleTopLevel(few)).toBeInstanceOf(Map);
  });


  it("gives up on streams with too many opcodes", () => {
    const many = Buffer.concat([PROTO2, EMPTY_LIST, ...Array.from({ length: 3_000_000 }, () => NONE), STOP]);
    expect(readPickleTopLevel(many)).toBeNull();
  });
});
