/*
 * myspin.rs - A handy program for testing your tiny shell
 *
 * usage: myspin <n>
 * Sleeps for <n> seconds in 1-second chunks.
 */

use std::env;
use std::process;
use std::thread;
use std::time::Duration;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <n>", args[0]);
        process::exit(0);
    }

    let secs = args[1].parse::<u64>().unwrap_or_else(|_| {
        eprintln!("Error: <n> must be a positive integer");
        process::exit(1);
    });

    for _ in 0..secs {
        thread::sleep(Duration::from_secs(1));
    }

    process::exit(0);
}
