#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct GeometryStream<'a> {
    mesh: &'a MeshBuffer,
}

#[cfg(windows)]
impl<'a> GeometryStream<'a> {
    pub fn from_mesh(mesh: &'a MeshBuffer) -> Self {
        Self { mesh }
    }

    pub fn mesh(&self) -> &'a MeshBuffer {
        self.mesh
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub struct InstanceStream<'a> {
    instances: &'a InstanceBuffer,
}

#[cfg(windows)]
impl<'a> InstanceStream<'a> {
    pub fn from_instances(instances: &'a InstanceBuffer) -> Self {
        Self { instances }
    }

    pub fn instances(&self) -> &'a InstanceBuffer {
        self.instances
    }
}
