//! OpenAI-shape error response helpers.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidRequest,
    Authentication,
    NotFound,
    RateLimit,
    Server,
}

impl ErrorKind {
    fn type_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request_error",
            ErrorKind::Authentication => "authentication_error",
            ErrorKind::NotFound => "not_found_error",
            ErrorKind::RateLimit => "rate_limit_error",
            ErrorKind::Server => "server_error",
        }
    }

    fn status(self) -> StatusCode {
        match self {
            ErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            ErrorKind::Authentication => StatusCode::UNAUTHORIZED,
            ErrorKind::NotFound => StatusCode::NOT_FOUND,
            ErrorKind::RateLimit => StatusCode::TOO_MANY_REQUESTS,
            ErrorKind::Server => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIError {
    pub kind: ErrorKind,
    pub message: String,
}

impl OpenAIError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, message)
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Server, message)
    }
}

impl std::fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind.type_str(), self.message)
    }
}

impl std::error::Error for OpenAIError {}

pub fn error_body(kind: ErrorKind, message: impl AsRef<str>) -> serde_json::Value {
    json!({
        "error": {
            "message": message.as_ref(),
            "type": kind.type_str(),
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    })
}

impl IntoResponse for OpenAIError {
    fn into_response(self) -> Response {
        let body = error_body(self.kind, &self.message);
        (self.kind.status(), Json(body)).into_response()
    }
}
