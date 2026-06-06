//! HTTP error plumbing + ID helpers shared across route handlers.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Which API's error envelope to render. The two surfaces disagree on
/// shape: OpenAI nests `{error:{message,type,code,param}}`; Anthropic uses
/// `{type:"error",error:{type,message}}`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ErrorFlavor {
    OpenAi,
    Anthropic,
}

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub flavor: ErrorFlavor,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            flavor: ErrorFlavor::OpenAi,
        }
    }
    pub fn queue_full() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "server queue full — too many concurrent requests".into(),
            flavor: ErrorFlavor::OpenAi,
        }
    }
    pub fn internal(err: impl std::fmt::Display) -> Self {
        let chain = format!("{err:#}");
        let top = err.to_string();
        tracing::error!(
            target: "server.api.error",
            top = %top,
            chain = %chain,
            "ApiError::internal",
        );
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: top,
            flavor: ErrorFlavor::OpenAi,
        }
    }

    /// Render this error in the Anthropic `/v1/messages` envelope.
    pub fn anthropic(mut self) -> Self {
        self.flavor = ErrorFlavor::Anthropic;
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let is_overloaded = self.status == StatusCode::SERVICE_UNAVAILABLE;
        let body = match self.flavor {
            ErrorFlavor::OpenAi => {
                let err_type = if is_overloaded {
                    "overloaded"
                } else {
                    "invalid_request_error"
                };
                json!({
                    "error": {
                        "message": self.message,
                        "type": err_type,
                        "code": serde_json::Value::Null,
                        "param": serde_json::Value::Null,
                    }
                })
            }
            ErrorFlavor::Anthropic => {
                let err_type = if is_overloaded {
                    "overloaded_error"
                } else if self.status == StatusCode::INTERNAL_SERVER_ERROR {
                    "api_error"
                } else {
                    "invalid_request_error"
                };
                json!({
                    "type": "error",
                    "error": {"type": err_type, "message": self.message},
                })
            }
        };
        let mut resp = (self.status, Json(body)).into_response();
        if is_overloaded {
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("2"),
            );
        }
        resp
    }
}

/// 503 response with `Retry-After: 2` for handlers that return `Response` directly
/// rather than `Result<_, ApiError>`.
pub fn queue_full_response() -> Response {
    let body = Json(json!({
        "error": {
            "message": "server queue full — too many concurrent requests",
            "type": "overloaded",
            "code": "queue_full"
        }
    }));
    let mut resp = (StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::RETRY_AFTER,
        axum::http::HeaderValue::from_static("2"),
    );
    resp
}

pub fn request_id(prefix: &str) -> String {
    format!("{prefix}-{}", now_unix())
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
