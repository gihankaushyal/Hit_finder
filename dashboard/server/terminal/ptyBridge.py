#!/usr/bin/env python3
"""Runs a command on a pseudo-terminal and relays bytes over stdin/stdout.

Usage: ptyBridge.py [--cols N] [--rows N] -- command [args...]

stdin carries frames: 1 type byte, a 4-byte big-endian length, then the payload.
  b"d"  keystrokes, written to the terminal as they are
  b"r"  resize, payload is cols and rows as two big-endian 16-bit numbers
Anything else is skipped. stdout carries the terminal's raw output bytes.

When stdin closes the command is hung up (SIGHUP, then SIGKILL after a moment) and the
bridge exits; when the command exits the bridge exits with its status. Standard library
only, and written to run on Python 3.6.
"""

import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios

HEADER = 5
MAX_FRAME = 1 << 20
MAX_BUFFERED = 4 << 20  # stop reading input while this much is waiting for the command
READ_CHUNK = 65536
HANGUP_WAIT_SECONDS = 1.0
DEFAULT_COLS = 80
DEFAULT_ROWS = 24


def set_size(fd, cols, rows):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def parse_args(argv):
    cols, rows = DEFAULT_COLS, DEFAULT_ROWS
    i = 0
    while i < len(argv) and argv[i] != "--":
        if argv[i] == "--cols" and i + 1 < len(argv):
            cols = int(argv[i + 1])
            i += 2
        elif argv[i] == "--rows" and i + 1 < len(argv):
            rows = int(argv[i + 1])
            i += 2
        else:
            raise SystemExit(
                "usage: ptyBridge.py [--cols N] [--rows N] -- command [args...]"
            )
    command = argv[i + 1 :]
    if not command:
        raise SystemExit(
            "usage: ptyBridge.py [--cols N] [--rows N] -- command [args...]"
        )
    return cols, rows, command


def write_all(fd, data):
    view = memoryview(data)
    while len(view):
        try:
            n = os.write(fd, view)
        except OSError as err:
            if err.errno == errno.EINTR:
                continue
            raise
        view = view[n:]


def exit_status(raw):
    if os.WIFEXITED(raw):
        return os.WEXITSTATUS(raw)
    if os.WIFSIGNALED(raw):
        return 128 + os.WTERMSIG(raw)
    return 1


class Terminate(Exception):
    """Raised by the signal handlers to leave the main loop and clean up."""


def on_signal(signum, frame):
    raise Terminate(signum)


def signal_group(pid, sig):
    # The command is a session leader (pty.fork), so its pid is also its process group.
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def hang_up(pid):
    signal_group(pid, signal.SIGHUP)
    import time

    end = time.time() + HANGUP_WAIT_SECONDS
    while time.time() < end:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except OSError:
            return
        if done:
            return
        time.sleep(0.02)
    signal_group(pid, signal.SIGKILL)
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def main(argv):
    cols, rows, command = parse_args(argv)
    # The parent sets the size and then releases the child, so the child never starts before the size is
    # right and can never overwrite a later resize.
    go_read, go_write = os.pipe()
    pid, master = pty.fork()
    if pid == 0:
        try:
            os.close(go_write)
            os.read(go_read, 1)
            os.close(go_read)
            os.execvp(command[0], command)
        except Exception as err:  # the child must never fall back into the parent's loop
            sys.stderr.write("cannot start %s: %s\n" % (command[0], err))
        os._exit(127)

    os.close(go_read)
    set_size(master, cols, rows)
    os.write(go_write, b"x")
    os.close(go_write)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    os.set_blocking(master, False)
    try:
        return relay(pid, master)
    except Terminate:
        hang_up(pid)
        return 143
    except Exception:
        # A closed output pipe, a vanished child or any other failure: never leave the command running.
        hang_up(pid)
        return 1


def relay(pid, master):
    pending = b""
    to_child = bytearray()
    stdin_open = True
    out_fd = sys.stdout.fileno()

    while True:
        readable = [master]
        if stdin_open and len(to_child) < MAX_BUFFERED:
            readable.append(0)
        writable = [master] if to_child else []
        ready_r, ready_w, _ = select.select(readable, writable, [])

        if master in ready_w and to_child:
            try:
                written = os.write(master, to_child)
                del to_child[:written]
            except BlockingIOError:
                pass
            except OSError as err:
                if err.errno != errno.EIO:
                    raise
                to_child = bytearray()

        if master in ready_r:
            try:
                chunk = os.read(master, READ_CHUNK)
            except BlockingIOError:
                chunk = None
            except OSError as err:
                if err.errno != errno.EIO:
                    raise
                chunk = b""
            if chunk is not None:
                if not chunk:
                    break  # the command closed its terminal
                write_all(out_fd, chunk)

        if stdin_open and 0 in ready_r:
            chunk = os.read(0, READ_CHUNK)
            if not chunk:
                hang_up(pid)
                return 0
            pending += chunk
            while len(pending) >= HEADER:
                kind = pending[0:1]
                length = struct.unpack(">I", pending[1:HEADER])[0]
                if length > MAX_FRAME:
                    hang_up(pid)
                    return 2
                if len(pending) < HEADER + length:
                    break
                payload = pending[HEADER : HEADER + length]
                pending = pending[HEADER + length :]
                if kind == b"d" and payload:
                    to_child += payload
                elif kind == b"r" and length == 4:
                    new_cols, new_rows = struct.unpack(">HH", payload)
                    if new_cols > 0 and new_rows > 0:
                        set_size(master, new_cols, new_rows)

    try:
        _, raw = os.waitpid(pid, 0)
    except OSError:
        return 0
    return exit_status(raw)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
