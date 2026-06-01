//! HTTP error plumbing + ID helpers shared across route handlers.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    pub fn queue_full() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "server queue full — too many concurrent requests".into(),
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
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let is_overloaded = self.status == StatusCode::SERVICE_UNAVAILABLE;
        let err_type = if is_overloaded {
            "overloaded"
        } else {
            "invalid_request_error"
        };
        let mut resp = (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": err_type,
                }
            })),
        )
            .into_response();
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
