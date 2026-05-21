/// Where a `forward_one_token_*` call should write its F32 logits row.
pub enum LogitsSink<'a> {
    Host(&'a mut Vec<f32>),
    Argmax,
    KeepOnDevice,
}

impl<'a> LogitsSink<'a> {
    pub fn is_host(&self) -> bool {
        matches!(self, LogitsSink::Host(_))
    }
    pub fn is_argmax(&self) -> bool {
        matches!(self, LogitsSink::Argmax)
    }
    pub fn is_keep_on_device(&self) -> bool {
        matches!(self, LogitsSink::KeepOnDevice)
    }
}
