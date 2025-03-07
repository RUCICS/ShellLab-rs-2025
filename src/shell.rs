use crate::builtins::handle_builtin;
use crate::exec::execute_command;
use crate::jobs::init_jobs;
use crate::parser::parse_command_line;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

/// Global prompt string.
pub static PROMPT: &str = "tsh> ";

/// Runs the main shell loop: prints the prompt (if enabled), reads input,
/// parses it, and evaluates commands.
///
/// - `emit_prompt`: if true, prints the command prompt.
/// - `verbose`: if true, prints extra debug information.
pub fn run_shell(emit_prompt: bool, verbose: bool) {
    // Create a shared job list protected by a mutex.
    let job_list = Arc::new(Mutex::new(init_jobs()));

    loop {
        if emit_prompt {
            print!("{}", PROMPT);
            io::stdout().flush().unwrap();
        }

        let mut cmdline = String::new();
        match io::stdin().read_line(&mut cmdline) {
            Ok(0) => break, // End-of-file (Ctrl-D)
            Ok(_) => {
                if cmdline.trim().is_empty() {
                    continue;
                }
                if verbose {
                    println!("Received command: {}", cmdline.trim());
                }
                // Parse the command line.
                match parse_command_line(&cmdline) {
                    Ok((command, bg)) => {
                        // If the command is built-in, execute it.
                        if handle_builtin(&command, &job_list) {
                            continue;
                        } else {
                            // Otherwise, execute it as an external command.
                            execute_command(command, bg, &job_list);
                        }
                    }
                    Err(e) => eprintln!("Parse error: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }
}
