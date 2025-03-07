# Rust shell lab Makefile
# Uses cargo for building the Rust project

# Default target - builds only the main program
all: build

# Build main program only
build:
	cargo build --bin tsh
	cp target/debug/tsh .

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
	cp target/release/myint .

myspin:
	cargo build --release --bin myspin
	cp target/release/myspin .

mysplit:
	cargo build --release --bin mysplit
	cp target/release/mysplit .

mystop:
	cargo build --release --bin mystop
	cp target/release/mystop .

.PHONY: all build build-all release run test clean aux myint myspin mysplit mystop
