#!/bin/sh
# Stand-in for the login shell in browser tests: no tmux, no profile, nothing of the user's.
printf 'fake-shell ready\n$ '
while IFS= read -r line; do
  case "$line" in
    exit) exit 0 ;;
  esac
  printf 'ran: %s\n$ ' "$line"
done
