#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialVertexRequirement {
    VisibleInstanceStream,
    DirectInstanceId,
    InstancePosition,
    GeometryPosition,
    ClipPositionOutput,
    VertexColorOutput,
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialFragmentRequirement {
    InterpolatedColorInput,
}
