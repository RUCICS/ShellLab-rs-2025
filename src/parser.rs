pub const MAXARGS: usize = 128;

/// Represents a parsed command.
#[derive(Debug)]
pub struct Command {
    /// Command and its arguments.
    pub argv: Vec<String>,
    /// Input redirection file, if any.
    pub infile: Option<String>,
    /// Output redirection file, if any.
    pub outfile: Option<String>,
    /// Append mode flag for output redirection.
    pub append: bool,
    /// Next command in a pipeline, if any.
    pub next: Option<Box<Command>>,
}

impl Command {
    /// Creates a new, empty command.
    pub fn new() -> Self {
        Command {
            argv: Vec::new(),
            infile: None,
            outfile: None,
            append: false,
            next: None,
        }
    }
}

/// Parses the input command line and returns a Command structure representing
/// the command (and pipeline, if present) along with a background execution flag.
/// This function handles:
///
/// - Tokenization (including quoted strings)
/// - Input redirection ("<") and output redirection (">" or ">>")
/// - A single pipeline ("|") by chaining commands
/// - Background execution using "&"
/// - For each token, calling `substitute_token()` to process environment variable
///   expansion and command substitution.
///
/// Returns `Ok((Command, bool))` on success, where the bool indicates background execution,
/// or `Err(String)` on error.
pub fn parse_command_line(cmdline: &str) -> Result<(Command, bool), String> {
    let tokens = tokenize(cmdline);
    if tokens.is_empty() {
        return Err("Empty command line".into());
    }

    let mut head = Command::new();
    let mut current = &mut head;
    let mut iter = tokens.into_iter().peekable();
    let mut bg = false;

    while let Some(token) = iter.next() {
        match token.as_str() {
            "<" => {
                if let Some(file_token) = iter.next() {
                    current.infile = Some(substitute_token(&file_token));
                } else {
                    return Err("No input file specified".into());
                }
            }
            ">" | ">>" => {
                let is_append = token == ">>";
                if let Some(file_token) = iter.next() {
                    current.outfile = Some(substitute_token(&file_token));
                    current.append = is_append;
                } else {
                    return Err("No output file specified".into());
                }
            }
            "|" => {
                let new_cmd = Command::new();
                current.next = Some(Box::new(new_cmd));
                if let Some(ref mut next_cmd) = current.next {
                    current = next_cmd;
                }
            }
            "&" => {
                bg = true;
            }
            _ => {
                if current.argv.len() < MAXARGS - 1 {
                    current.argv.push(substitute_token(&token));
                } else {
                    return Err("Too many arguments".into());
                }
            }
        }
    }
    Ok((head, bg))
}

/// Splits the input command line into a vector of tokens. This function handles:
///
/// - Quoted strings (using single or double quotes)
/// - Special tokens: `<`, `>`, `>>`, `|`, and `&`
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        // Handle quoted tokens.
        if ch == '"' || ch == '\'' {
            let quote = ch;
            chars.next(); // Consume opening quote.
            let mut token = String::new();
            while let Some(&c) = chars.peek() {
                if c == quote {
                    chars.next(); // Consume closing quote.
                    break;
                } else {
                    token.push(c);
                    chars.next();
                }
            }
            tokens.push(token);
        }
        // Handle special tokens.
        else if ch == '<' || ch == '>' || ch == '|' || ch == '&' {
            if ch == '>' {
                chars.next();
                if let Some(&next_ch) = chars.peek() {
                    if next_ch == '>' {
                        chars.next();
                        tokens.push(">>".to_string());
                        continue;
                    } else {
                        tokens.push(">".to_string());
                        continue;
                    }
                }
            } else {
                tokens.push(ch.to_string());
                chars.next();
            }
        }
        // Normal unquoted token.
        else {
            let mut token = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '<' || c == '>' || c == '|' || c == '&' {
                    break;
                }
                token.push(c);
                chars.next();
            }
            tokens.push(token);
        }
    }
    tokens
}

/// Processes a token for environment variable expansion and command substitution.
/// If the token starts with "$(" and ends with ")", it calls `command_substitute()`.
/// If it starts with '$', it calls `env_expand()`. Otherwise, it returns the token unchanged.
fn substitute_token(token: &str) -> String {
    if token.starts_with("$(") && token.ends_with(")") {
        return command_substitute(token);
    }
    if token.starts_with('$') {
        return env_expand(token);
    }
    token.to_string()
}

/// Stub for environment variable expansion.
/// TODO: Implement environment variable expansion.
fn env_expand(token: &str) -> String {
    // For now, simply return the token unchanged.
    token.to_string()
}

/// Stub for command substitution.
/// TODO: Implement command substitution.
fn command_substitute(token: &str) -> String {
    // For now, simply return the token unchanged.
    token.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tokenize_simple() {
        let input = "ls -l";
        let tokens = tokenize(input);
        assert_eq!(tokens, vec!["ls", "-l"]);
    }
    #[test]
    fn test_tokenize_quotes() {
        let input = "echo \"hello world\"";
        let tokens = tokenize(input);
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }
    #[test]
    fn test_parse_command_line() {
        let input = "grep 'pattern' < input.txt | sort > output.txt &";
        let (cmd, bg) = parse_command_line(input).unwrap();
        assert_eq!(cmd.argv, vec!["grep", "pattern"]);
        assert_eq!(cmd.infile, Some("input.txt".to_string()));
        let next_cmd = cmd.next.unwrap();
        assert_eq!(next_cmd.argv, vec!["sort"]);
        assert_eq!(next_cmd.outfile, Some("output.txt".to_string()));
        assert!(bg);
    }
}
