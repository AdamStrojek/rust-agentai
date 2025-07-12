//! # Built-in Tools
//!
//! This module provides a collection of commonly used tools that can be shared across different AI agents.
//! These tools offer functionalities for common tasks such as retrieving the current date and time, and fetching location data.
//!
//! ## Included Toolboxes:
//!
//! - `CurrentDateAndTimeToolBox`: A set of tools for querying the current date, time, and performing timezone conversions.
//! - `LocationToolBox`: A tool for retrieving geographical information (latitude and longitude) for a given location using the OpenStreetMap Nominatim API.
//!
//! For a practical demonstration of how to use these tools, please refer to the `examples/tool_buildin.rs` file.
use crate::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
use anyhow::anyhow;
use time::format_description::well_known::{Iso8601, Rfc3339};
use time::{format_description, Date, OffsetDateTime, Time};
use time_tz::{timezones, OffsetDateTimeExt};

/// # Current Date and Time Toolbox
///
/// This struct provides tools for getting the current date and time.
/// The `#[toolbox]` macro exposes methods marked with `#[tool]` to an AI model,
/// enabling it to answer questions about the current date and time.
pub struct CurrentDateAndTimeToolBox {}

#[toolbox]
impl CurrentDateAndTimeToolBox {
    /// Use this tool to answer questions like: "What is today's date?".
    /// It returns the date in `YYYY-MM-DD` format.
    /// The date is based on the local timezone of the system.
    #[tool]
    pub fn get_today_date(&self) -> ToolResult {
        let today = OffsetDateTime::now_local().map_err(|err| ToolError::Other(anyhow!(err)))?;
        today
            .date()
            .format(&Iso8601::DATE)
            .map_err(|err| ToolError::Other(anyhow!(err)))
    }

    /// Use this tool to answer questions like: "What time is it?".
    /// It returns the time in `HH:MM:SS` format.
    /// The time is based on the local timezone of the system.
    #[tool]
    pub fn get_current_time(&self) -> ToolResult {
        let now = OffsetDateTime::now_local().map_err(|e| ToolError::Other(anyhow!(e)))?;
        let format = format_description::parse("[hour]:[minute]:[second]")
            .map_err(|e| ToolError::Other(anyhow!(e)))?;
        now.format(&format)
            .map_err(|e| ToolError::Other(anyhow!(e)))
    }

    /// Use this tool to get the complete current date and time for precise and unambiguous time-stamping.
    /// For example, to answer "What is the current timestamp?".
    /// Returns a timestamp in the standard ISO 8601 format (e.g., "2023-10-27T10:30:00+00:00").
    #[tool]
    pub fn get_current_datetime(&self) -> ToolResult {
        let now = OffsetDateTime::now_local().map_err(|e| ToolError::Other(anyhow!(e)))?;
        now.format(&Rfc3339)
            .map_err(|e| ToolError::Other(anyhow!(e)))
        // Ok(now.to_string())
    }

    /// Use this tool to find the day of the week for a given date. For example, to answer "What day of the week was 2024-01-01?".
    #[tool]
    pub fn get_day_of_week(
        &self,
        /// Date in `YYYY-MM-DD` format
        date: String,
    ) -> ToolResult {
        let parsed_date =
            Date::parse(&date, &Iso8601::DEFAULT).map_err(|err| ToolError::Other(anyhow!(err)))?;
        Ok(parsed_date.weekday().to_string())
    }

    /// Use this tool to answer questions like: "What time is it in Tokyo?".
    /// You must provide the timezone as a string
    /// It returns the time in `HH:MM:SS` format for that zone.
    #[tool]
    pub fn get_time_in_timezone(
        &self,
        /// Timezone provided in IANA timezone names format (e.g., "America/New_York", "Europe/London", "Asia/Tokyo").
        timezone: String,
    ) -> ToolResult {
        let tz = timezones::get_by_name(&timezone)
            .ok_or_else(|| ToolError::Other(anyhow!("Unknown timezone: {}", timezone)))?;
        let now_utc = OffsetDateTime::now_utc();
        let now_in_tz = now_utc.to_timezone(tz);
        let format = format_description::parse("[hour]:[minute]:[second]")
            .map_err(|e| ToolError::Other(anyhow!(e)))?;
        now_in_tz
            .format(&format)
            .map_err(|e| ToolError::Other(anyhow!(e)))
    }

    /// Use this tool to convert time between different timezones. For example, to answer "What is 14:00 in New York in Tokyo time?".
    /// You must provide the source timezone, the time to convert, and the target timezone.
    #[tool]
    pub fn convert_time(
        &self,
        /// Source timezone provided in IANA timezone names format (e.g., "America/New_York", "Asia/Tokyo").
        source_timezone: String,
        /// Time in `HH:MM` format to be converted
        time: String,
        /// Target timezone provided in IANA timezone names format (e.g., "America/New_York", "Asia/Tokyo").
        target_timezone: String,
    ) -> ToolResult {
        let source_tz = timezones::get_by_name(&source_timezone).ok_or_else(|| {
            ToolError::Other(anyhow!("Unknown source timezone: {}", source_timezone))
        })?;
        let target_tz = timezones::get_by_name(&target_timezone).ok_or_else(|| {
            ToolError::Other(anyhow!("Unknown target timezone: {}", target_timezone))
        })?;

        let time_format = format_description::parse("[hour]:[minute]")
            .map_err(|e| ToolError::Other(anyhow!(e)))?;
        let parsed_time = Time::parse(&time, &time_format)
            .map_err(|e| ToolError::Other(anyhow!("Invalid time format for '{}': {}", time, e)))?;

        let now_in_source_tz = OffsetDateTime::now_utc().to_timezone(source_tz);
        let source_datetime = now_in_source_tz.replace_time(parsed_time);

        let target_datetime = source_datetime.to_timezone(target_tz);

        target_datetime
            .format(&time_format)
            .map_err(|e| ToolError::Other(anyhow!(e)))
    }
}

#[derive(serde::Deserialize)]
struct LocationResponse {
    display_name: Box<str>,
    lat: Box<str>,
    lon: Box<str>,
}

/// # Location Toolbox
///
/// This struct provides tools for getting location data from the Nominatim OpenStreetMap API.
///
/// Many times others tools like weather API accepts only geolocation as coordinates. Humans do not
/// communicate that way, we prefer to provide proper address. By utilizing location search we can
/// allow to provide it as address, which will be converted to geolocation.
///
/// Please remember to follow Nominatim Usage Policy
/// <https://operations.osmfoundation.org/policies/nominatim/>
pub struct LocationToolBox;

#[toolbox]
impl LocationToolBox {
    /// Use this tool to get the geographical location (latitude and longitude) of a place.
    /// For example, to answer "Where is the Eiffel Tower?". You can search using not only a city name
    /// but also more specific details, like a full street address.
    /// It returns the display name, latitude, and longitude.
    #[tool]
    pub async fn get_location(
        &self,
        /// The name of the location to search for (e.g., "Eiffel Tower", "New York City").
        location: String,
    ) -> ToolResult {
        let url = format!("https://nominatim.openstreetmap.org/search?q={location}&format=jsonv2");

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            // Nominatim API requires a User-Agent header.
            .header("User-Agent", "rust-agentai-client")
            .send()
            .await
            .map_err(|e| ToolError::Other(anyhow!("Failed to send request: {}", e)))?;

        if !response.status().is_success() {
            // TODO: Unify way of API Error
            return Err(ToolError::Other(anyhow!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let locations: Vec<LocationResponse> = response
            .json()
            .await
            .map_err(|e| ToolError::Other(anyhow!("Failed to parse JSON response: {}", e)))?;

        if let Some(first_location) = locations.first() {
            Ok(format!(
                "Location: {}, Latitude: {}, Longitude: {}",
                first_location.display_name, first_location.lat, first_location.lon
            ))
        } else {
            Err(ToolError::Other(anyhow!(
                "No location found for '{}'",
                location
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::format_description::well_known::Iso8601;
    use time::{Date, OffsetDateTime};

    #[test]
    fn test_get_today_date() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_today_date().unwrap();
        assert!(Date::parse(&result, &Iso8601::DATE).is_ok());
    }

    #[test]
    fn test_get_current_time() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_current_time().unwrap();
        let parts: Vec<&str> = result.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 2);
        assert_eq!(parts[2].len(), 2);
    }

    #[test]
    fn test_get_day_of_week() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_day_of_week("2024-01-01".to_string()).unwrap();
        assert_eq!(result, "Monday");
    }

    #[test]
    fn test_get_day_of_week_invalid_date() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_day_of_week("invalid-date".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_current_datetime() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_current_datetime().unwrap();
        assert!(OffsetDateTime::parse(&result, &Iso8601::DEFAULT).is_ok());
    }

    #[test]
    fn test_get_time_in_timezone() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox
            .get_time_in_timezone("Asia/Tokyo".to_string())
            .unwrap();
        let parts: Vec<&str> = result.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 2);
        assert_eq!(parts[2].len(), 2);
    }

    #[test]
    fn test_get_time_in_invalid_timezone() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.get_time_in_timezone("Invalid/Timezone".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_time() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox
            .convert_time(
                "America/New_York".to_string(),
                "10:00".to_string(),
                "Asia/Tokyo".to_string(),
            )
            .unwrap();
        let parts: Vec<&str> = result.split(':').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 2);
    }

    #[test]
    fn test_convert_time_invalid_input() {
        let toolbox = CurrentDateAndTimeToolBox {};
        let result = toolbox.convert_time(
            "Invalid/Timezone".to_string(),
            "10:00".to_string(),
            "Asia/Tokyo".to_string(),
        );
        assert!(result.is_err());

        let result = toolbox.convert_time(
            "America/New_York".to_string(),
            "99:99".to_string(),
            "Asia/Tokyo".to_string(),
        );
        assert!(result.is_err());

        let result = toolbox.convert_time(
            "America/New_York".to_string(),
            "10:00".to_string(),
            "Invalid/Timezone".to_string(),
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_location() {
        let toolbox = LocationToolBox;
        let result = toolbox.get_location("Wrocław".to_string()).await;
        assert!(result.is_ok());
        let location_info = result.unwrap();
        assert!(location_info.contains("Location: Wrocław"));
        assert!(location_info.contains("Latitude: 51."));
        assert!(location_info.contains("Longitude: 16."));
    }

    #[tokio::test]
    async fn test_get_location_not_found() {
        let toolbox = LocationToolBox;
        let result = toolbox
            .get_location("SomeInvalidPlaceThatDoesNotExist".to_string())
            .await;
        assert!(result.is_err());
    }
}
