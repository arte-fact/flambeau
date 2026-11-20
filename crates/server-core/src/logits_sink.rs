/// Where a `forward_one_token_*` call should write its F32 logits row.
pub enum LogitsSink<'a> {
    Host(&'a mut Vec<f32>),
    Argmax,
    KeepOnDevice,
}
