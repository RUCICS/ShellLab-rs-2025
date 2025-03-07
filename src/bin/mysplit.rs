/*
 * mysplit.rs - Another handy routine for testing your tiny shell
 *
 * usage: mysplit <n>
 * Fork a child that spins for <n> seconds in 1-second chunks.
 */

use nix::sys::wait::wait;
use nix::unistd::{fork, ForkResult};
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

    match unsafe { fork() } {
        Ok(ForkResult::Child) => {
            // Child process
            for _ in 0..secs {
                thread::sleep(Duration::from_secs(1));
            }
            process::exit(0);
        }
        Ok(ForkResult::Parent { .. }) => {
            // Parent process waits for child to terminate
            if let Err(err) = wait() {
                eprintln!("wait error: {}", err);
            }
            process::exit(0);
        }
        Err(err) => {
            eprintln!("fork error: {}", err);
            process::exit(1);
        }
    }
}
