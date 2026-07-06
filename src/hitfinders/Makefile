# src/hitfinders/Makefile
# Compile the CrystFEL PF8 C bridge for use by PF8Hitfinder.
#
# Usage (from repo root or src/hitfinders/):
#   cd src/hitfinders && make
#
# Output: src/hitfinders/_pf8_wrap.so
#
# Requires: gcc, CrystFEL at CRYSTFEL_ROOT

CRYSTFEL_ROOT ?= /data/bioxfel/software/crystfel-0.12.0
CONDA_ENV     ?= /home/gketawal/.conda/envs/sfx-hitfinder

CC      = gcc
CFLAGS  = -O2 -fPIC -Wall \
          -I$(CRYSTFEL_ROOT)/include \
          -I$(CONDA_ENV)/include
LDFLAGS = -shared \
          -L$(CRYSTFEL_ROOT)/lib64 \
          -L$(CONDA_ENV)/lib \
          -lcrystfel -lgsl -lgslcblas \
          -Wl,-rpath,$(CRYSTFEL_ROOT)/lib64 \
          -Wl,-rpath,$(CONDA_ENV)/lib

TARGET  = _pf8_wrap.so
SRC     = _pf8_wrap.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)
