use std::fmt;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub layouts: Vec<LayoutDecl>,
    pub kernels: Vec<Kernel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDecl {
    pub name: String,
    pub kind: LayoutKind,
    pub fields: Vec<LayoutField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutKind {
    AoS,
    SoA,
    AoSoA { group_size: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutField {
    pub name: String,
    pub ty: TypeName,
    pub semantic: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub kind: EntryPointKind,
    pub name: String,
    pub params: Vec<Param>,
    pub body: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPointKind {
    Kernel,
    Vertex,
    Fragment,
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
    U8x4Unorm,
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
    #[error("graphics lowering requires at least one `{0} fn` entrypoint")]
    MissingGraphicsStage(&'static str),
    #[error("graphics lowering could not find `{kind} fn {name}`")]
    MissingNamedGraphicsStage { kind: &'static str, name: String },
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
        if kernel.kind != EntryPointKind::Kernel {
            continue;
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphicsShaders {
    pub vertex_source: String,
    pub fragment_source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HlslRegister {
    pub register: u32,
    pub space: u32,
}

impl HlslRegister {
    pub const fn new(register: u32, space: u32) -> Self {
        Self { register, space }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphicsBindings {
    pub raster_params: HlslRegister,
    pub visible_instances: HlslRegister,
    pub instances: HlslRegister,
    pub geometry: HlslRegister,
}

impl Default for GraphicsBindings {
    fn default() -> Self {
        Self {
            raster_params: HlslRegister::new(0, 0),
            visible_instances: HlslRegister::new(0, 0),
            instances: HlslRegister::new(1, 0),
            geometry: HlslRegister::new(2, 0),
        }
    }
}

pub fn lower_graphics_to_hlsl(source: &str) -> Result<GraphicsShaders, LowerError> {
    let program = parse(source)?;
    lower_graphics_program_to_hlsl(&program)
}

pub fn lower_graphics_to_hlsl_with_bindings(
    source: &str,
    bindings: GraphicsBindings,
) -> Result<GraphicsShaders, LowerError> {
    let program = parse(source)?;
    lower_graphics_program_to_hlsl_with_bindings(&program, bindings)
}

pub fn lower_graphics_to_hlsl_for_entries_with_bindings(
    source: &str,
    vertex_entrypoint: &str,
    fragment_entrypoint: &str,
    bindings: GraphicsBindings,
) -> Result<GraphicsShaders, LowerError> {
    let program = parse(source)?;
    lower_graphics_program_to_hlsl_for_entries_with_bindings(
        &program,
        vertex_entrypoint,
        fragment_entrypoint,
        bindings,
    )
}

pub fn lower_graphics_program_to_hlsl(program: &Program) -> Result<GraphicsShaders, LowerError> {
    lower_graphics_program_to_hlsl_with_bindings(program, GraphicsBindings::default())
}

pub fn lower_graphics_program_to_hlsl_with_bindings(
    program: &Program,
    bindings: GraphicsBindings,
) -> Result<GraphicsShaders, LowerError> {
    let vertex = program
        .kernels
        .iter()
        .find(|entry| entry.kind == EntryPointKind::Vertex)
        .ok_or(LowerError::MissingGraphicsStage("vertex"))?;
    let fragment = program
        .kernels
        .iter()
        .find(|entry| entry.kind == EntryPointKind::Fragment)
        .ok_or(LowerError::MissingGraphicsStage("fragment"))?;
    Ok(GraphicsShaders {
        vertex_source: hlsl_stage(vertex, bindings),
        fragment_source: hlsl_stage(fragment, bindings),
    })
}

pub fn lower_graphics_program_to_hlsl_for_entries_with_bindings(
    program: &Program,
    vertex_entrypoint: &str,
    fragment_entrypoint: &str,
    bindings: GraphicsBindings,
) -> Result<GraphicsShaders, LowerError> {
    let vertex = named_graphics_stage(program, EntryPointKind::Vertex, vertex_entrypoint)?;
    let fragment = named_graphics_stage(program, EntryPointKind::Fragment, fragment_entrypoint)?;
    Ok(GraphicsShaders {
        vertex_source: hlsl_stage(vertex, bindings),
        fragment_source: hlsl_stage(fragment, bindings),
    })
}

fn named_graphics_stage<'a>(
    program: &'a Program,
    kind: EntryPointKind,
    name: &str,
) -> Result<&'a Kernel, LowerError> {
    let label = match kind {
        EntryPointKind::Kernel => "kernel",
        EntryPointKind::Vertex => "vertex",
        EntryPointKind::Fragment => "fragment",
    };
    program
        .kernels
        .iter()
        .find(|entry| entry.kind == kind && entry.name == name)
        .ok_or_else(|| LowerError::MissingNamedGraphicsStage {
            kind: label,
            name: name.to_string(),
        })
}

fn hlsl_stage(entry: &Kernel, bindings: GraphicsBindings) -> String {
    let stage = match entry.kind {
        EntryPointKind::Kernel => "compute",
        EntryPointKind::Vertex => "vertex",
        EntryPointKind::Fragment => "fragment",
    };
    let mut out = format!("// Neo {stage} stage: {}\n", entry.name);
    out.push_str(&hlsl_raster_prelude(bindings));
    match entry.kind {
        EntryPointKind::Vertex => {
            out.push_str("NeoVertexOut ");
            out.push_str(&entry.name);
            out.push_str("(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {\n");
            out.push_str("    NeoVertexOut outp;\n");
            out.push_str("    outp.position = float4(0.0, 0.0, 0.0, 1.0);\n");
            out.push_str("    outp.color = float4(1.0, 1.0, 1.0, 1.0);\n");
        }
        EntryPointKind::Fragment => {
            out.push_str("float4 ");
            out.push_str(&entry.name);
            out.push_str("(NeoVertexOut input) : SV_Target {\n");
        }
        EntryPointKind::Kernel => {
            out.push_str("void ");
            out.push_str(&entry.name);
            out.push_str("() {\n");
        }
    }
    out.push_str(&rewrite_hlsl_body(&entry.body));
    if !entry.body.ends_with('\n') {
        out.push('\n');
    }
    match entry.kind {
        EntryPointKind::Vertex => out.push_str("    return outp;\n"),
        EntryPointKind::Fragment => out.push_str("    return input.color;\n"),
        EntryPointKind::Kernel => {}
    }
    out.push_str("}\n");
    out
}

fn rewrite_hlsl_body(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        out.push_str(&rewrite_hlsl_line(line));
        out.push('\n');
    }
    out.replace("vertex_id()", "vertex_id")
        .replace("instance_id()", "instance_id")
        .replace("input_color()", "input.color")
        .replace("vec2f(", "float2(")
        .replace("vec3f(", "float3(")
        .replace("vec4f(", "float4(")
}

fn rewrite_hlsl_line(line: &str) -> String {
    if let Some(rewritten) = rewrite_hlsl_setter(line, "set_position", "outp.position") {
        return rewritten;
    }
    if let Some(rewritten) = rewrite_hlsl_setter(line, "set_color", "outp.color") {
        return rewritten;
    }
    rewrite_let_line(line, hlsl_type_name).unwrap_or_else(|| line.to_string())
}

fn rewrite_hlsl_setter(line: &str, neo_name: &str, hlsl_name: &str) -> Option<String> {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    let trimmed = rest.trim();
    let prefix = format!("{neo_name}(");
    if !trimmed.starts_with(&prefix) || !trimmed.ends_with(");") {
        return None;
    }
    let expr = &trimmed[prefix.len()..trimmed.len() - 2];
    Some(format!("{indent}{hlsl_name} = {expr};"))
}

fn hlsl_raster_prelude(bindings: GraphicsBindings) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\ncbuffer RasterParams : register(b{}, space{})\n{{\n",
        bindings.raster_params.register, bindings.raster_params.space
    ));
    out.push_str(HLSL_RASTER_PARAMS_BODY);
    out.push_str(&format!(
        "\nByteAddressBuffer neo_visible_instances : register(t{}, space{});\n",
        bindings.visible_instances.register, bindings.visible_instances.space
    ));
    out.push_str(&format!(
        "ByteAddressBuffer neo_instances : register(t{}, space{});\n",
        bindings.instances.register, bindings.instances.space
    ));
    out.push_str(&format!(
        "ByteAddressBuffer neo_geometry : register(t{}, space{});\n",
        bindings.geometry.register, bindings.geometry.space
    ));
    out.push_str(HLSL_RASTER_PRELUDE_BODY);
    out
}

const HLSL_RASTER_PARAMS_BODY: &str = r#"    uint neo_grid_x;
    uint neo_grid_y;
    uint neo_grid_z;
    uint neo_frame;
    uint neo_width;
    uint neo_height;
    uint neo_reserved0;
    uint neo_reserved1;
    float4 neo_camera_origin;
    float4 neo_camera_right;
    float4 neo_camera_up;
    float4 neo_camera_forward;
    float4 neo_camera_view;
};
"#;

const HLSL_RASTER_PRELUDE_BODY: &str = r#"
struct NeoVertexOut {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

uint visible_instance_id(uint draw_instance) { return neo_visible_instances.Load(draw_instance * 4); }
uint neo_geometry_stride() { return max(neo_reserved0, 16); }
uint neo_geometry_color_offset() { return neo_reserved1; }
float3 neo_geometry_position3f(uint vertex_id)
{
    uint bytes = vertex_id * neo_geometry_stride();
    return float3(asfloat(neo_geometry.Load(bytes)), asfloat(neo_geometry.Load(bytes + 4)), asfloat(neo_geometry.Load(bytes + 8)));
}
uint neo_geometry_color4u8(uint vertex_id)
{
    return neo_geometry.Load(vertex_id * neo_geometry_stride() + neo_geometry_color_offset());
}
uint neo_instance_header_u32(uint byte_offset) { return neo_instances.Load(byte_offset); }
uint neo_instance_count() { return neo_instance_header_u32(12); }
uint neo_instance_stride() { return neo_instance_header_u32(16); }
uint neo_instance_data_offset() { return neo_instance_header_u32(20); }
uint neo_instance_attr_count() { return neo_instance_header_u32(24); }
uint neo_instance_attr_offset() { return neo_instance_header_u32(28); }
uint neo_instance_layout_kind() { return neo_instance_header_u32(32); }
uint neo_instance_group_size() { uint size = neo_instance_header_u32(36); return size == 0 ? 1 : size; }
uint neo_instance_format_size(uint format)
{
    if (format == 1) return 8;
    if (format == 2) return 12;
    if (format == 3) return 16;
    if (format == 4) return 4;
    return 0;
}
uint neo_instance_find_attr(uint semantic)
{
    uint attrs = neo_instance_attr_offset();
    uint count = neo_instance_attr_count();
    [loop]
    for (uint i = 0; i < count; ++i) {
        uint attr = attrs + i * 16;
        if (neo_instances.Load(attr) == semantic) {
            return attr;
        }
    }
    return 0xffffffffu;
}
uint neo_instance_attr_bytes(uint semantic, uint instance_id, out uint format)
{
    uint attr = neo_instance_find_attr(semantic);
    if (attr == 0xffffffffu || instance_id >= neo_instance_count()) {
        format = 0;
        return 0;
    }
    format = neo_instances.Load(attr + 4);
    uint attr_offset = neo_instances.Load(attr + 8);
    uint layout = neo_instance_layout_kind();
    uint data = neo_instance_data_offset();
    if (layout == 1) {
        return data + attr_offset + instance_id * neo_instance_format_size(format);
    }
    if (layout == 2) {
        uint group_size = neo_instance_group_size();
        uint group = instance_id / group_size;
        uint lane = instance_id - group * group_size;
        uint element_size = neo_instance_format_size(format);
        return data + attr_offset + group * group_size * element_size + lane * element_size;
    }
    return data + instance_id * neo_instance_stride() + attr_offset;
}
float3 neo_instance_position3f(uint instance_id)
{
    uint format;
    uint bytes = neo_instance_attr_bytes(1, instance_id, format);
    if (format == 1) return float3(asfloat(neo_instances.Load(bytes)), asfloat(neo_instances.Load(bytes + 4)), 0.0);
    if (format == 2 || format == 3) return float3(asfloat(neo_instances.Load(bytes)), asfloat(neo_instances.Load(bytes + 4)), asfloat(neo_instances.Load(bytes + 8)));
    return float3(0.0, 0.0, 0.0);
}
float2 neo_instance_scale2f(uint instance_id)
{
    uint format;
    uint bytes = neo_instance_attr_bytes(3, instance_id, format);
    if (format == 1 || format == 2 || format == 3) return float2(asfloat(neo_instances.Load(bytes)), asfloat(neo_instances.Load(bytes + 4)));
    return float2(1.0, 1.0);
}
uint neo_instance_color4u8(uint instance_id)
{
    uint format;
    uint bytes = neo_instance_attr_bytes(4, instance_id, format);
    if (format == 4) return neo_instances.Load(bytes);
    return 0xffffffffu;
}
uint neo_stress_instance_group_size()
{
    return neo_instance_group_size();
}
uint neo_stress_instance_group_slots()
{
    uint count = neo_instance_count();
    uint group_size = neo_stress_instance_group_size();
    return ((count + group_size - 1) / group_size) * group_size;
}
uint neo_stress_instance_aosoa_base(uint instance_id, uint element_size)
{
    uint group_size = neo_stress_instance_group_size();
    uint group = instance_id / group_size;
    uint lane = instance_id - group * group_size;
    return group * group_size * element_size + lane * element_size;
}
float3 neo_stress_instance_position3f(uint instance_id)
{
    uint bytes = neo_instance_data_offset() + neo_stress_instance_aosoa_base(instance_id, 12);
    return float3(asfloat(neo_instances.Load(bytes)), asfloat(neo_instances.Load(bytes + 4)), asfloat(neo_instances.Load(bytes + 8)));
}
float2 neo_stress_instance_scale2f(uint instance_id)
{
    uint slots = neo_stress_instance_group_slots();
    uint stream = slots * 12 + slots * 16;
    uint bytes = neo_instance_data_offset() + stream + neo_stress_instance_aosoa_base(instance_id, 8);
    return float2(asfloat(neo_instances.Load(bytes)), asfloat(neo_instances.Load(bytes + 4)));
}
uint neo_stress_instance_color4u8(uint instance_id)
{
    uint slots = neo_stress_instance_group_slots();
    uint stream = slots * 12 + slots * 16 + slots * 8;
    return neo_instances.Load(neo_instance_data_offset() + stream + neo_stress_instance_aosoa_base(instance_id, 4));
}
float4 unpack_bgra8(uint packed)
{
    return float4(
        (float)((packed >> 16) & 255) / 255.0,
        (float)((packed >> 8) & 255) / 255.0,
        (float)(packed & 255) / 255.0,
        (float)((packed >> 24) & 255) / 255.0);
}
uint raster_grid_x() { return neo_grid_x; }
uint raster_grid_y() { return neo_grid_y; }
uint raster_grid_z() { return neo_grid_z; }
uint raster_frame() { return neo_frame; }
uint raster_width() { return neo_width; }
uint raster_height() { return neo_height; }
float raster_aspect() { return neo_height == 0 ? 1.0 : ((float)neo_width / (float)neo_height); }
float3 raster_camera_origin() { return neo_camera_origin.xyz; }
float3 raster_camera_right() { return neo_camera_right.xyz; }
float3 raster_camera_up() { return neo_camera_up.xyz; }
float3 raster_camera_forward() { return neo_camera_forward.xyz; }
float raster_camera_tan_x() { return max(neo_camera_view.x, 0.001); }
float raster_camera_tan_y() { return max(neo_camera_view.y, 0.001); }
float raster_instance_spacing() { return neo_camera_view.z; }
float as_f32(uint x) { return (float)x; }
float as_f32(int x) { return (float)x; }
float as_f32(float x) { return x; }
uint as_u32(uint x) { return x; }
uint as_u32(int x) { return (uint)x; }
uint as_u32(float x) { return (uint)x; }

"#;

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

fn hlsl_type_name(ty: &TypeName) -> &'static str {
    match ty {
        TypeName::Bool => "bool",
        TypeName::I32 => "int",
        TypeName::U8 => "uint",
        TypeName::U32 => "uint",
        TypeName::F32 => "float",
        TypeName::Vec2f => "float2",
        TypeName::Vec3f => "float3",
        TypeName::Vec4f => "float4",
        TypeName::U8x4Unorm => "uint",
    }
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
        TypeName::U8x4Unorm => "unsigned int",
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
    rewrite_let_line(line, cuda_type_name).unwrap_or_else(|| line.to_string())
}

fn rewrite_let_line(line: &str, type_name: fn(&TypeName) -> &'static str) -> Option<String> {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    let trimmed = rest.trim_start();
    if !trimmed.starts_with("let ") {
        return None;
    }
    let colon = trimmed.find(':')?;
    let eq = trimmed.find('=')?;
    if colon > eq {
        return None;
    }

    let name = trimmed[4..colon].trim();
    let ty = trimmed[colon + 1..eq].trim();
    let expr = trimmed[eq + 1..].trim();

    let lowered_ty = lower_decl_type_text(ty, type_name);
    Some(format!("{indent}{lowered_ty} {name} = {expr}"))
}

fn lower_decl_type_text(text: &str, type_name: fn(&TypeName) -> &'static str) -> String {
    let mut ty = text.trim();
    let mut pointer_depth = 0usize;
    while let Some(stripped) = ty.strip_suffix('*') {
        pointer_depth += 1;
        ty = stripped.trim_end();
    }

    let mut out = match parse_type_name_text(ty) {
        Some(ty) => type_name(&ty).to_string(),
        None => ty.to_string(),
    };
    for _ in 0..pointer_depth {
        out.push('*');
    }
    out
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
        "u8x4_unorm" => Some(TypeName::U8x4Unorm),
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
            if "(){}[],:;*.+-/<>=%!&|@?^".contains(ch) {
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
        let mut layouts = Vec::new();
        let mut kernels = Vec::new();
        while !self.is_eof() {
            match self.peek_ident() {
                Some("layout") => layouts.push(self.parse_layout()?),
                Some("kernel") => kernels.push(self.parse_entrypoint(EntryPointKind::Kernel)?),
                Some("vertex") => kernels.push(self.parse_entrypoint(EntryPointKind::Vertex)?),
                Some("fragment") => kernels.push(self.parse_entrypoint(EntryPointKind::Fragment)?),
                _ => {
                    let token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
                    return Err(ParseError {
                        diagnostic: Diagnostic::new(
                            "expected `layout`, `kernel`, `vertex`, or `fragment`",
                            token.span,
                        ),
                    });
                }
            }
        }
        Ok(Program { layouts, kernels })
    }

    fn parse_layout(&mut self) -> Result<LayoutDecl, ParseError> {
        let start = self.expect_ident_text("layout")?.span.start;
        let kind_token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
        let kind_name = self.expect_ident()?;
        let kind = match kind_name.as_str() {
            "aos" => LayoutKind::AoS,
            "soa" => LayoutKind::SoA,
            "aosoa" => {
                self.expect_symbol('(')?;
                let number_token = self.peek_token().cloned().ok_or_else(|| self.eof_error())?;
                let group_size = self
                    .expect_number()?
                    .parse::<u32>()
                    .map_err(|_| ParseError {
                        diagnostic: Diagnostic::new(
                            "expected integer AoSoA group size",
                            number_token.span,
                        ),
                    })?;
                self.expect_symbol(')')?;
                LayoutKind::AoSoA { group_size }
            }
            _ => {
                return Err(ParseError {
                    diagnostic: Diagnostic::new(
                        "expected layout kind `aos`, `soa`, or `aosoa`",
                        kind_token.span,
                    ),
                });
            }
        };
        let name = self.expect_ident()?;
        self.expect_symbol('{')?;
        let mut fields = Vec::new();
        while !self.check_symbol('}') {
            let field_name = self.expect_ident()?;
            self.expect_symbol(':')?;
            let ty = self.parse_type_name()?;
            self.expect_symbol('@')?;
            let semantic = self.expect_ident()?;
            self.eat_symbol(',');
            fields.push(LayoutField {
                name: field_name,
                ty,
                semantic,
            });
        }
        let close = self.expect_symbol('}')?;
        Ok(LayoutDecl {
            name,
            kind,
            fields,
            span: Span {
                start,
                end: close.span.end,
            },
        })
    }

    fn parse_entrypoint(&mut self, kind: EntryPointKind) -> Result<Kernel, ParseError> {
        let keyword = match kind {
            EntryPointKind::Kernel => "kernel",
            EntryPointKind::Vertex => "vertex",
            EntryPointKind::Fragment => "fragment",
        };
        let start = self.expect_ident_text(keyword)?.span.start;
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
            kind,
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

    fn expect_number(&mut self) -> Result<String, ParseError> {
        let token = self.peek_token().ok_or_else(|| self.eof_error())?;
        match &token.kind {
            TokenKind::Number(value) => {
                let value = value.clone();
                self.idx += 1;
                Ok(value)
            }
            _ => Err(ParseError {
                diagnostic: Diagnostic::new("expected number", token.span),
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
        assert!(program.layouts.is_empty());
        assert_eq!(program.kernels.len(), 1);
        let kernel = &program.kernels[0];
        assert_eq!(kernel.kind, EntryPointKind::Kernel);
        assert_eq!(kernel.name, "image");
        assert_eq!(kernel.params[0].address_space, Some(AddressSpace::Global));
        assert_eq!(kernel.params[0].ty.base, TypeName::U8);
        assert_eq!(kernel.params[0].ty.pointer_depth, 1);
    }

    #[test]
    fn parses_vertex_and_fragment_entrypoints() {
        let program =
            parse("vertex fn quad_vs() { let id: u32 = vertex_id(); }\nfragment fn quad_fs() { }")
                .unwrap();
        assert_eq!(program.kernels.len(), 2);
        assert_eq!(program.kernels[0].kind, EntryPointKind::Vertex);
        assert_eq!(program.kernels[0].name, "quad_vs");
        assert_eq!(program.kernels[1].kind, EntryPointKind::Fragment);
        assert_eq!(program.kernels[1].name, "quad_fs");
    }

    #[test]
    fn parses_layout_declaration() {
        let program = parse(
            "layout aosoa(32) Instance {\n    position: vec3f @ Position,\n    color: u8x4_unorm @ Color0,\n}\nkernel fn image(global u8* pixels) {}",
        )
        .unwrap();
        assert_eq!(program.layouts.len(), 1);
        assert_eq!(program.layouts[0].name, "Instance");
        assert_eq!(
            program.layouts[0].kind,
            LayoutKind::AoSoA { group_size: 32 }
        );
        assert_eq!(program.layouts[0].fields[0].ty, TypeName::Vec3f);
        assert_eq!(program.layouts[0].fields[1].ty, TypeName::U8x4Unorm);
        assert_eq!(program.kernels.len(), 1);
    }

    #[test]
    fn reports_invalid_layout_syntax_with_span() {
        let err = parse("layout chunky Instance {}").unwrap_err();
        assert!(err.diagnostic.message.contains("expected layout kind"));
        assert!(err.diagnostic.span.end > err.diagnostic.span.start);
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
            "kernel fn image(global u8* pixels, u32 width) {\n    let x: u32 = thread_id().x;\n}\nvertex fn ignored() {}",
        )
        .unwrap();
        assert!(cuda.contains("extern \"C\" __global__ void image"));
        assert!(!cuda.contains("void ignored"));
        assert!(cuda.contains("unsigned char* pixels"));
        assert!(cuda.contains("unsigned int x = threadIdx.x;"));
    }

    #[test]
    fn lowers_pointer_and_custom_let_types_to_cuda_declarations() {
        let cuda = lower_to_cuda(
            "kernel fn image(global u8* pixels, global u8* camera) {\n    let cam: NeoStressCamera* = (NeoStressCamera*)camera;\n    let data: unsigned char* = pixels;\n    let values: float* = (float*)data;\n    let typed: u8* = pixels;\n}",
        )
        .unwrap();
        assert!(cuda.contains("NeoStressCamera* cam = (NeoStressCamera*)camera;"));
        assert!(cuda.contains("unsigned char* data = pixels;"));
        assert!(cuda.contains("float* values = (float*)data;"));
        assert!(cuda.contains("unsigned char* typed = pixels;"));
    }

    #[test]
    fn lowers_graphics_entrypoints_to_hlsl() {
        let hlsl = lower_graphics_to_hlsl(
            "kernel fn cull(global u8* args) {}\nvertex fn quad_vs() {\n    let id: u32 = vertex_id();\n    set_position(vec4f(1.0f, 0.0f, 0.0f, 1.0f));\n    set_color(vec4f(0.0f, 1.0f, 0.0f, 1.0f));\n}\nfragment fn quad_fs() {\n    return input_color();\n}",
        )
        .unwrap();
        assert!(hlsl.vertex_source.contains("Neo vertex stage: quad_vs"));
        assert!(hlsl.vertex_source.contains("SV_VertexID"));
        assert!(hlsl.vertex_source.contains("vertex_id"));
        assert!(hlsl.vertex_source.contains("visible_instance_id"));
        assert!(hlsl.vertex_source.contains("raster_camera_origin"));
        assert!(hlsl.vertex_source.contains("raster_camera_forward"));
        assert!(hlsl.vertex_source.contains("outp.position = float4"));
        assert!(hlsl.vertex_source.contains("outp.color = float4"));
        assert!(hlsl.fragment_source.contains("Neo fragment stage: quad_fs"));
        assert!(hlsl.fragment_source.contains("SV_Target"));
        assert!(hlsl.fragment_source.contains("return input.color;"));
    }

    #[test]
    fn lowers_named_graphics_entrypoints_to_hlsl() {
        let source = r#"
vertex fn fallback_vs() {
    set_position(vec4f(0.0f, 0.0f, 0.0f, 1.0f));
}
fragment fn fallback_fs() {
    return input_color();
}
vertex fn material_vs() {
    let id: u32 = visible_instance_id(instance_id());
    set_position(vec4f(as_f32(id), 0.0f, 0.0f, 1.0f));
}
fragment fn material_fs() {
    return input_color();
}
"#;
        let hlsl = lower_graphics_to_hlsl_for_entries_with_bindings(
            source,
            "material_vs",
            "material_fs",
            GraphicsBindings::default(),
        )
        .unwrap();
        assert!(hlsl.vertex_source.contains("Neo vertex stage: material_vs"));
        assert!(hlsl.vertex_source.contains("material_vs("));
        assert!(!hlsl.vertex_source.contains("fallback_vs("));
        assert!(
            hlsl.fragment_source
                .contains("Neo fragment stage: material_fs")
        );
        assert!(hlsl.fragment_source.contains("material_fs("));
        assert!(!hlsl.fragment_source.contains("fallback_fs("));

        let err = lower_graphics_to_hlsl_for_entries_with_bindings(
            source,
            "missing_vs",
            "material_fs",
            GraphicsBindings::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("vertex fn missing_vs"));
    }

    #[test]
    fn lowers_graphics_entrypoints_with_explicit_hlsl_bindings() {
        let hlsl = lower_graphics_to_hlsl_with_bindings(
            "vertex fn quad_vs() {\n    set_position(vec4f(0.0f, 0.0f, 0.0f, 1.0f));\n}\nfragment fn quad_fs() {\n    return input_color();\n}",
            GraphicsBindings {
                raster_params: HlslRegister::new(4, 2),
                visible_instances: HlslRegister::new(7, 3),
                instances: HlslRegister::new(8, 3),
                geometry: HlslRegister::new(9, 4),
            },
        )
        .unwrap();
        assert!(
            hlsl.vertex_source
                .contains("cbuffer RasterParams : register(b4, space2)")
        );
        assert!(
            hlsl.vertex_source
                .contains("neo_visible_instances : register(t7, space3)")
        );
        assert!(
            hlsl.vertex_source
                .contains("neo_instances : register(t8, space3)")
        );
        assert!(
            hlsl.vertex_source
                .contains("neo_geometry : register(t9, space4)")
        );
        assert!(
            hlsl.fragment_source
                .contains("cbuffer RasterParams : register(b4, space2)")
        );
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
