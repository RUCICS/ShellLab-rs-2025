use std::collections::HashMap;

/// Represents the state of a job.
#[derive(Debug, PartialEq, Eq)]
pub enum JobState {
    Undefined,
    Foreground,
    Background,
    Stopped,
}

/// Represents a job in the shell.
#[derive(Debug)]
pub struct Job {
    pub pid: i32,
    pub jid: i32,
    pub state: JobState,
    pub cmdline: String,
}

/// Manages the list of jobs using a HashMap keyed by process ID.
pub struct JobList {
    pub jobs: HashMap<i32, Job>,
    pub next_jid: i32,
}

impl JobList {
    /// Creates a new, empty job list.
    pub fn new() -> Self {
        JobList {
            jobs: HashMap::new(),
            next_jid: 1,
        }
    }
}

/// Initializes and returns a new job list.
pub fn init_jobs() -> JobList {
    JobList::new()
}

/// Returns the maximum job ID currently in the job list.
pub fn max_jid(job_list: &JobList) -> i32 {
    job_list.jobs.values().map(|job| job.jid).max().unwrap_or(0)
}

/// Adds a new job to the job list.
pub fn add_job(job_list: &mut JobList, pid: i32, state: JobState, cmdline: String) -> bool {
    if pid < 1 {
        return false;
    }
    let jid = job_list.next_jid;
    let job = Job {
        pid,
        jid,
        state,
        cmdline,
    };
    job_list.jobs.insert(pid, job);
    job_list.next_jid += 1;
    true
}

/// Deletes the job with the given pid from the job list.
pub fn delete_job(job_list: &mut JobList, pid: i32) -> bool {
    if pid < 1 {
        return false;
    }
    let removed = job_list.jobs.remove(&pid);
    if removed.is_some() {
        job_list.next_jid = max_jid(job_list) + 1;
        true
    } else {
        false
    }
}

/// Returns the process ID of the foreground job, if any.
pub fn fg_pid(job_list: &JobList) -> Option<i32> {
    for job in job_list.jobs.values() {
        if job.state == JobState::Foreground {
            return Some(job.pid);
        }
    }
    None
}

/// Returns a reference to the job with the given pid.
pub fn get_job_by_pid<'a>(job_list: &'a JobList, pid: i32) -> Option<&'a Job> {
    job_list.jobs.get(&pid)
}

/// Returns a reference to the job with the given job ID.
pub fn get_job_by_jid<'a>(job_list: &'a JobList, jid: i32) -> Option<&'a Job> {
    job_list.jobs.values().find(|job| job.jid == jid)
}

/// Maps a process ID to its job ID.
pub fn pid_to_jid(job_list: &JobList, pid: i32) -> Option<i32> {
    job_list.jobs.get(&pid).map(|job| job.jid)
}

/// Prints the list of jobs.
pub fn list_jobs(job_list: &JobList) {
    for job in job_list.jobs.values() {
        let state_str = match job.state {
            JobState::Background => "Running",
            JobState::Foreground => "Foreground",
            JobState::Stopped => "Stopped",
            JobState::Undefined => "Undefined",
        };
        println!("[{}] ({}) {} {}", job.jid, job.pid, state_str, job.cmdline);
    }
}
