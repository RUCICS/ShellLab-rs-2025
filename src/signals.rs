use signal_hook::{consts::signal::*, iterator::Signals};
use std::thread;

/// Installs signal handlers for the shell, including:
/// - SIGQUIT: Prints a termination message and exits.
/// - SIGINT: (Ctrl-C) Should forward the signal to the foreground job (TODO).
/// - SIGTSTP: (Ctrl-Z) Should stop the foreground job (TODO).
/// - SIGCHLD: Should reap zombie processes (TODO).
pub fn install_signal_handlers() {
    let mut signals = Signals::new(&[SIGQUIT, SIGINT, SIGTSTP, SIGCHLD])
        .expect("Unable to create signal handler");
    thread::spawn(move || {
        for signal in signals.forever() {
            match signal {
                SIGQUIT => {
                    println!("Terminating after receipt of SIGQUIT signal");
                    std::process::exit(0);
                }
                SIGINT => {
                    // TODO: Forward SIGINT (Ctrl-C) to the foreground job.
                    println!("Received SIGINT (Ctrl-C)"); // Placeholder
                }
                SIGTSTP => {
                    // TODO: Stop the foreground job.
                    println!("Received SIGTSTP (Ctrl-Z)"); // Placeholder
                }
                SIGCHLD => {
                    // TODO: Reap zombie processes.
                    println!("Received SIGCHLD"); // Placeholder
                }
                _ => unreachable!(),
            }
        }
    });
}
