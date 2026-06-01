use std::fmt;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub kernels: Vec<Kernel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Param>,
    pub body: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub address_space: Option<AddressSpace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub base: TypeName,
    pub pointer_depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeName {
    Bool,
    I32,
    U8,
    U32,
    F32,
    Vec2f,
    Vec3f,
    Vec4f,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
}

impl Diagnostic {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at byte {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{diagnostic}")]
pub struct ParseError {
    pub diagnostic: Diagnostic,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LowerError {
    #[error("{0}")]
    Parse(#[from] ParseError),
}

pub fn parse(source: &str) -> Result<Program, ParseError> {
    Parser::new(source).parse_program()
}

pub fn lower_to_cuda(source: &str) -> Result<String, LowerError> {
    let program = parse(source)?;
    Ok(lower_program(&program))
}

pub fn lower_program(program: &Program) -> String {
    let mut out = String::from(
        r#"#define as_u8(x) ((unsigned char)(x))
#define as_i32(x) ((int)(x))
#define as_u32(x) ((unsigned int)(x))
#define as_f32(x) ((float)(x))

"#,
    );

    for kernel in &program.kernels {
        out.push_str("extern \"C\" __global__ void ");
        out.push_str(&kernel.name);
        out.push('(');
        for (idx, param) in kernel.params.iter().enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            out.push_str(&cuda_param(param));
        }
        out.push_str(") {\n");
        out.push_str(&rewrite_body(&kernel.body));
        if !kernel.body.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("}\n\n");
    }

    out
}

fn cuda_param(param: &Param) -> String {
    let mut out = String::new();
    if matches!(param.address_space, Some(AddressSpace::Shared)) {
        out.push_str("__shared__ ");
    }
    out.push_str(cuda_type_name(&param.ty.base));
    for _ in 0..param.ty.pointer_depth {
        out.push('*');
    }
    out.push(' ');
    out.push_str(&param.name);
    out
}

fn cuda_type_name(ty: &TypeName) -> &'static str {
    match ty {
        TypeName::Bool => "bool",
        TypeName::I32 => "int",
        TypeName::U8 => "unsigned char",
        TypeName::U32 => "unsigned int",
        TypeName::F32 => "float",
        TypeName::Vec2f => "float2",
        TypeName::Vec3f => "float3",
        TypeName::Vec4f => "float4",
    }
}

fn rewrite_body(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        out.push_str(&rewrite_line(line));
        out.push('\n');
    }

    out.replace("thread_id()", "threadIdx")
        .replace("block_id()", "blockIdx")
        .replace("block_dim()", "blockDim")
        .replace("grid_dim()", "gridDim")
        .replace("block_barrier()", "__syncthreads()")
        .replace("vec2f(", "make_float2(")
        .replace("vec3f(", "make_float3(")
        .replace("vec4f(", "make_float4(")
}

fn rewrite_line(line: &str) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    let trimmed = rest.trim_start();
    if let Some(shared) = trimmed.strip_prefix("shared ") {
        return rewrite_shared_line(indent, shared);
    }
    if !trimmed.starts_with("let ") {
        return line.to_string();
    }

    let Some(colon) = trimmed.find(':') else {
        return line.to_string();
    };
    let Some(eq) = trimmed.find('=') else {
        return line.to_string();
    };
    if colon > eq {
        return line.to_string();
    }

    let name = trimmed[4..colon].trim();
    let ty = trimmed[colon + 1..eq].trim();
    let expr = trimmed[eq + 1..].trim();

    let cuda_ty = match parse_type_name_text(ty) {
        Some(ty) => cuda_type_name(&ty),
        None => ty,
    };

    format!("{indent}{cuda_ty} {name} = {expr}")
}

fn rewrite_shared_line(indent: &str, shared: &str) -> String {
    let shared = shared.trim_start();
    let Some(split) = shared.find(char::is_whitespace) else {
        return format!("{indent}shared {shared}");
    };
    let (ty, rest) = shared.split_at(split);
    let cuda_ty = match parse_type_name_text(ty) {
        Some(ty) => cuda_type_name(&ty),
        None => ty,
    };
    format!("{indent}__shared__ {cuda_ty} {}", rest.trim_start())
}

fn parse_type_name_text(text: &str) -> Option<TypeName> {
    match text.trim() {
        "bool" => Some(TypeName::Bool),
        "i32" => Some(TypeName::I32),
        "u8" => Some(TypeName::U8),
        "u32" => Some(TypeName::U32),
        "f32" => Some(TypeName::F32),
        "vec2f" => Some(TypeName::Vec2f),
        "vec3f" => Some(TypeName::Vec3f),
        "vec4f" => Some(TypeName::Vec4f),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenKind {
    Ident(String),
    Number(String),
    Symbol(char),
}

struct Lexer<'a> {
    source: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, pos: 0 }
    }

    fn lex(mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.bump();
                continue;
            }
            if ch == '/' && self.peek_next() == Some('/') {
                self.bump();
                self.bump();
                while let Some(ch) = self.peek() {
                    self.bump();
                    if ch == '\n' {
                        break;
                    }
                }
                continue;
            }
            let start = self.pos;
            if is_ident_start(ch) {
                self.bump();
                while self.peek().is_some_and(is_ident_continue) {
                    self.bump();
                }
                tokens.push(Token {
                    kind: TokenKind::Ident(self.source[start..self.pos].to_string()),
                    span: Span {
                        start,
                        end: self.pos,
                    },
                });
                continue;
            }
            if ch.is_ascii_digit() {
                self.bump();
                while self
                    .peek()
                    .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '.')
                {
                    self.bump();
                }
                tokens.push(Token {
                    kind: TokenKind::Number(self.source[start..self.pos].to_string()),
                    span: Span {
                        start,
                        end: self.pos,
                    },
                });
                continue;
            }
            if "(){}[],:;*.+-/<>=%!&|".contains(ch) {
                self.bump();
                tokens.push(Token {
                    kind: TokenKind::Symbol(ch),
                    span: Span {
                        start,
                        end: self.pos,
                    },
                });
                continue;
            }
            return Err(ParseError {
                diagnostic: Diagnostic::new(
                    format!("unexpected character `{ch}`"),
                    Span {
                        start,
                        end: start + ch.len_utf8(),
                    },
                ),
            });
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek_next(&self) -> Option<char> {
        let mut chars = self.source[self.pos..].chars();
        chars.next()?;
        chars.next()
    }

    fn bump(&mut self) {
        if let Some(ch) = self.peek() {
            self.pos += ch.len_utf8();
        }
    }
}

struct Parser<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    idx: usize,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            tokens: Vec::new(),
            idx: 0,
        }
    }

    fn parse_program(mut self) -> Result<Program, ParseError> {
        self.tokens = Lexer::new(self.source).lex()?;
        let mut kernels = Vec::new();
        while !self.is_eof() {
            kernels.push(self.parse_kernel()?);
        }
        Ok(Program { kernels })
    }

    fn parse_kernel(&mut self) -> Result<Kernel, ParseError> {
        let start = self.expect_ident_text("kernel")?.span.start;
        self.expect_ident_text("fn")?;
        let name = self.expect_ident()?;
        self.expect_symbol('(')?;

        let mut params = Vec::new();
        if !self.check_symbol(')') {
            loop {
                params.push(self.parse_param()?);
                if self.eat_symbol(',') {
                    continue;
                }
                break;
            }
        }
        self.expect_symbol(')')?;
        let open = self.expect_symbol('{')?;
        let close_idx = self.find_matching_brace(self.idx - 1)?;
        let close = self.tokens[close_idx].clone();
        let body = self.source[open.span.end..close.span.start].to_string();
        self.idx = close_idx + 1;

        Ok(Kernel {
            name,
            params,
            body,
            span: Span {
                start,
                end: close.span.end,
            },
        })
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let address_space = self.parse_address_space();
        let base = self.parse_type_name()?;
        let mut pointer_depth = 0;
        while self.eat_symbol('*') {
            pointer_depth += 1;
        }
        let name = self.expect_ident()?;
        Ok(Param {
            name,
            ty: Type {
                base,
                pointer_depth,
            },
            address_space,
        })
    }

    fn parse_address_space(&mut self) -> Option<AddressSpace> {
        let ident = self.peek_ident()?;
        let space = match ident {
            "global" => AddressSpace::Global,
            "shared" => AddressSpace::Shared,
            "local" => AddressSpace::Local,
            _ => return None,
        };
        self.idx += 1;
        Some(space)
    }

    fn parse_type_name(&mut self) -> Result<TypeName, ParseError> {
        let token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
        let ident = self.expect_ident()?;
        parse_type_name_text(&ident).ok_or_else(|| ParseError {
            diagnostic: Diagnostic::new(format!("unknown type `{ident}`"), token.span),
        })
    }

    fn find_matching_brace(&self, open_idx: usize) -> Result<usize, ParseError> {
        let mut depth = 0usize;
        for idx in open_idx..self.tokens.len() {
            match self.tokens[idx].kind {
                TokenKind::Symbol('{') => depth += 1,
                TokenKind::Symbol('}') => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(idx);
                    }
                }
                _ => {}
            }
        }
        Err(ParseError {
            diagnostic: Diagnostic::new("unclosed kernel body", self.tokens[open_idx].span),
        })
    }

    fn expect_ident_text(&mut self, expected: &str) -> Result<Token, ParseError> {
        let token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
        match &token.kind {
            TokenKind::Ident(value) if value == expected => {
                self.idx += 1;
                Ok(token)
            }
            _ => Err(ParseError {
                diagnostic: Diagnostic::new(format!("expected `{expected}`"), token.span),
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let token = self.peek_token().ok_or_else(|| self.eof_error())?;
        match &token.kind {
            TokenKind::Ident(value) => {
                let value = value.clone();
                self.idx += 1;
                Ok(value)
            }
            _ => Err(ParseError {
                diagnostic: Diagnostic::new("expected identifier", token.span),
            }),
        }
    }

    fn expect_symbol(&mut self, expected: char) -> Result<Token, ParseError> {
        let token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
        match token.kind {
            TokenKind::Symbol(value) if value == expected => {
                self.idx += 1;
                Ok(token)
            }
            _ => Err(ParseError {
                diagnostic: Diagnostic::new(format!("expected `{expected}`"), token.span),
            }),
        }
    }

    fn eat_symbol(&mut self, expected: char) -> bool {
        if self.check_symbol(expected) {
            self.idx += 1;
            true
        } else {
            false
        }
    }

    fn check_symbol(&self, expected: char) -> bool {
        matches!(
            self.peek_token().map(|token| &token.kind),
            Some(TokenKind::Symbol(value)) if *value == expected
        )
    }

    fn peek_ident(&self) -> Option<&str> {
        match self.peek_token().map(|token| &token.kind) {
            Some(TokenKind::Ident(value)) => Some(value),
            _ => None,
        }
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.idx)
    }

    fn is_eof(&self) -> bool {
        self.idx >= self.tokens.len()
    }

    fn eof_error(&self) -> ParseError {
        ParseError {
            diagnostic: Diagnostic::new(
                "unexpected end of file",
                Span {
                    start: self.source.len(),
                    end: self.source.len(),
                },
            ),
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    is_ident_start(ch) || ch.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_declaration_with_address_space() {
        let program = parse("kernel fn image(global u8* pixels, u32 width) {}").unwrap();
        assert_eq!(program.kernels.len(), 1);
        let kernel = &program.kernels[0];
        assert_eq!(kernel.name, "image");
        assert_eq!(kernel.params[0].address_space, Some(AddressSpace::Global));
        assert_eq!(kernel.params[0].ty.base, TypeName::U8);
        assert_eq!(kernel.params[0].ty.pointer_depth, 1);
    }

    #[test]
    fn reports_invalid_syntax_with_span() {
        let err = parse("kernel image() {}").unwrap_err();
        assert!(err.diagnostic.message.contains("expected `fn`"));
        assert!(err.diagnostic.span.end > err.diagnostic.span.start);
    }

    #[test]
    fn lowers_kernel_to_cuda() {
        let cuda = lower_to_cuda(
            "kernel fn image(global u8* pixels, u32 width) {\n    let x: u32 = thread_id().x;\n}",
        )
        .unwrap();
        assert!(cuda.contains("extern \"C\" __global__ void image"));
        assert!(cuda.contains("unsigned char* pixels"));
        assert!(cuda.contains("unsigned int x = threadIdx.x;"));
    }

    #[test]
    fn lowers_vector_literals() {
        let cuda = lower_to_cuda(
            "kernel fn shade(global u8* pixels) {\n    let color: vec4f = vec4f(1.0f, 0.0f, 0.0f, 1.0f);\n}",
        )
        .unwrap();
        assert!(cuda.contains("float4 color = make_float4"));
    }

    #[test]
    fn lowers_shared_locals_and_block_barrier() {
        let cuda = lower_to_cuda(
            "kernel fn tile(global u8* pixels) {\n    shared i32 tile_window[4];\n    tile_window[0] = 1;\n    block_barrier();\n}",
        )
        .unwrap();
        assert!(cuda.contains("__shared__ int tile_window[4];"));
        assert!(cuda.contains("__syncthreads();"));
    }
}
