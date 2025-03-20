import { goto } from "$app/navigation";

export function getBaseUrl() {
    if (typeof window !== "undefined") {
        const { protocol, host } = window.location;
        return `${protocol}//${host}/api/`;
    }
    return "";
}

export async function submitGetRequest(endpoint: string) {
    try {
        const response = await fetch(`${getBaseUrl()}${endpoint}`, {
            method: "GET",
            credentials: "include" // Optional: for cookies/auth
        });

        if (!response.ok) {
            if (response.status === 401) {
                console.log("Session expired. Redirecting to error page...");
                goto("/error");
                return null;
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("GET request failed:", error);
        throw error; // Re-throw to let the caller handle it
    }
}

export async function submitPostRequest(endpoint: string, data: any = {}) {
    try {
        const response = await fetch(`${getBaseUrl()}${endpoint}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data),
            credentials: "include" // Optional: for cookies/auth
        });

        if (!response.ok) {
            if (response.status === 401) {
                console.log("Session expired. Redirecting to error page...");
                goto("/error");
                return null;
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("POST request failed:", error);
        throw error; // Re-throw to let the caller handle it
    }
}