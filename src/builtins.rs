use crate::jobs::{list_jobs, JobList};
use crate::parser::Command;
use std::sync::{Arc, Mutex};

/// Checks if the command is a built-in command and, if so, executes it.
/// Supported built-ins include "quit", "jobs", "fg", "bg", and "cd".
/// Returns true if the command was built-in and handled; false otherwise.
pub fn handle_builtin(cmd: &Command, job_list: &Arc<Mutex<JobList>>) -> bool {
    if cmd.argv.is_empty() {
        return false;
    }
    match cmd.argv[0].as_str() {
        "quit" => {
            std::process::exit(0);
        }
        "jobs" => {
            let jl = job_list.lock().unwrap();
            list_jobs(&jl);
            return true;
        }
        "fg" | "bg" => {
            // TODO: Implement foreground/background job control.
            println!("{} command not implemented yet", cmd.argv[0]);
            return true;
        }
        "cd" => {
            // TODO: Implement changing directories.
            println!("cd command not implemented yet");
            return true;
        }
        _ => false,
    }
}
