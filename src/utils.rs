use std::process;

pub fn print_usage() {
    println!("Usage: shell [-hvp]");
    println!("   -h   Print this help message");
    println!("   -v   Enable verbose mode");
    println!("   -p   Do not print a command prompt");
    process::exit(1);
}

pub fn error(msg: &str) {
    eprintln!("Error: {}", msg);
    process::exit(1);
}
