use crate::jobs::JobList;
use crate::parser::Command;
use std::sync::{Arc, Mutex};

/// Executes an external command.
/// This function should handle:
/// - Forking a child process.
/// - Setting the child's process group (for job control).
/// - Handling I/O redirection (infile, outfile, append mode).
/// - Executing the command using exec (or similar).
/// - In the parent, adding the job to the job list if needed and waiting for foreground jobs.
///
/// For now, this function is a stub. Students should implement the actual process control logic.
pub fn execute_command(cmd: Command, job_list: &Arc<Mutex<JobList>>) {
    println!("Executing external command: {:?}", cmd);

    // TODO:
    // 1. Fork the process.
    // 2. In the child:
    //    - Set the process group (e.g., using nix::unistd::setpgid).
    //    - Handle redirections (infile, outfile, append mode).
    //    - Execute the command (e.g., using execvp).
    // 3. In the parent:
    //    - Add the job to the job list.
    //    - If the command is foreground, wait for it to finish.
}
