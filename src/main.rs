mod builtins;
mod exec;
mod jobs;
mod parser;
mod shell;
mod signals;
mod utils;

use std::env;

fn main() {
    // Parse command-line arguments.
    let args: Vec<String> = env::args().collect();
    let mut emit_prompt = true;
    let mut verbose = false;
    for arg in &args[1..] {
        match arg.as_str() {
            "-h" => utils::print_usage(),
            "-v" => verbose = true,
            "-p" => emit_prompt = false,
            _ => {}
        }
    }

    // Install signal handlers.
    signals::install_signal_handlers();

    // Run the main shell loop with the options.
    shell::run_shell(emit_prompt, verbose);
}
