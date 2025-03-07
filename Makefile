# Rust shell lab Makefile
# Uses cargo for building the Rust project

# Default target - builds only the main program
all: build

# Build main program only
build:
	cargo build --bin tsh
	@if [ ! -e tsh ]; then ln -s target/debug/tsh .; fi

# Build main program and auxiliary programs
build-all: build aux

# Build the main program in release mode
release:
	cargo build --release --bin tsh

# Run the shell
run:
	cargo run

# Run the tests
test:
	python3 grader.py

# Clean build artifacts
clean:
	cargo clean
	rm -f myint myspin mysplit mystop tsh

# Build all auxiliary programs (optimized for size)
aux: myint myspin mysplit mystop

# Individual auxiliary programs
myint:
	cargo build --release --bin myint
	@if [ ! -e myint ]; then ln -s target/release/myint .; fi

myspin:
	cargo build --release --bin myspin
	@if [ ! -e myspin ]; then ln -s target/release/myspin .; fi

mysplit:
	cargo build --release --bin mysplit
	@if [ ! -e mysplit ]; then ln -s target/release/mysplit .; fi

mystop:
	cargo build --release --bin mystop
	@if [ ! -e mystop ]; then ln -s target/release/mystop .; fi

.PHONY: all build build-all release run test clean aux myint myspin mysplit mystop
