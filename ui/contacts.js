const API_BASE = "http://127.0.0.1:8000/api";

// ── Auth Helper ──────────────────────────────────────────────────────────────
// ✅ Key matches script.js: localStorage.setItem("auth_token", ...)
function getAuthHeaders() {
    const token = localStorage.getItem("auth_token");
    if (!token) {
        window.location.href = "/";
        return null;
    }
    return {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
    };
}
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    loadContacts();

    const slider  = document.getElementById("contact-risk");
    const valText = document.getElementById("risk-val");
    if (slider) {
        slider.addEventListener("input", (e) => {
            valText.innerText = parseFloat(e.target.value).toFixed(2);
        });
    }

    const savedTheme = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);
});

function toggleRiskSlider() {
    const isTrusted = document.getElementById("contact-trusted").checked;
    const group   = document.getElementById("risk-score-group");
    const slider  = document.getElementById("contact-risk");
    const valText = document.getElementById("risk-val");

    if (isTrusted) {
        group.style.display   = "none";
        slider.value          = 0.0;
        valText.innerText     = "0.00";
    } else {
        group.style.display   = "block";
        slider.value          = 0.5;
        valText.innerText     = "0.50";
    }
}

async function loadContacts() {
    const deviceId = document.getElementById("device-select").value;
    const tbody    = document.getElementById("contacts-tbody");

    tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">
        <i class="fa-solid fa-spinner fa-spin"></i> Loading records...
    </td></tr>`;

    const headers = getAuthHeaders();
    if (!headers) return;

    try {
        const response = await fetch(`${API_BASE}/devices/${deviceId}/contacts`, {
            headers: headers
        });

        if (response.status === 401) {
            localStorage.removeItem("auth_token");
            window.location.href = "/";
            return;
        }

        const contacts = await response.json();

        if (contacts.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No contacts stored in this edge database.</td></tr>`;
            return;
        }

        tbody.innerHTML = contacts.map(c => `
            <tr>
                <td><code>${c.id}</code></td>
                <td><strong>${c.name}</strong></td>
                <td><code>${c.phone}</code></td>
                <td>
                    <span class="badge-status ${c.is_trusted ? "status-approved" : "status-blocked"}">
                        ${c.is_trusted ? "Trusted" : "Untrusted/Flagged"}
                    </span>
                </td>
                <td><strong class="${c.risk_score > 0.6 ? "text-amber" : ""}">${c.risk_score.toFixed(2)}</strong></td>
                <td>
                    <button class="btn btn-small btn-block" onclick="deleteContact(${c.id})">
                        <i class="fa-solid fa-trash-can"></i> Delete
                    </button>
                </td>
            </tr>
        `).join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-red">Failed to connect to backend server.</td></tr>`;
        console.error(err);
    }
}

async function addContact(event) {
    event.preventDefault();

    const deviceId   = document.getElementById("device-select").value;
    const name       = document.getElementById("contact-name").value.trim();
    const phone      = document.getElementById("contact-phone").value.trim();
    const is_trusted = document.getElementById("contact-trusted").checked;
    const risk_score = is_trusted ? 0.0 : parseFloat(document.getElementById("contact-risk").value);

    const headers = getAuthHeaders();
    if (!headers) return;

    try {
        const response = await fetch(`${API_BASE}/devices/${deviceId}/contacts`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify({ name, phone, is_trusted, risk_score })
        });

        if (response.status === 401) {
            localStorage.removeItem("auth_token");
            window.location.href = "/";
            return;
        }

        if (response.ok) {
            showToast(`Contact saved directly to SQLite DB contacts_${deviceId}.db!`, "success");
            document.getElementById("contact-form").reset();
            document.getElementById("contact-trusted").checked = true;
            toggleRiskSlider();
            loadContacts();
        } else {
            const err = await response.json();
            showToast(err.detail || "Error saving contact.", "error");
        }
    } catch (err) {
        showToast("Network error occurred.", "error");
        console.error(err);
    }
}

async function deleteContact(contactId) {
    const deviceId = document.getElementById("device-select").value;

    if (!confirm("Are you sure you want to delete this contact from the SQLite database?")) return;

    const headers = getAuthHeaders();
    if (!headers) return;

    try {
        const response = await fetch(`${API_BASE}/devices/${deviceId}/contacts/${contactId}`, {
            method: "DELETE",
            headers: headers
        });

        if (response.status === 401) {
            localStorage.removeItem("auth_token");
            window.location.href = "/";
            return;
        }

        if (response.ok) {
            showToast("Contact deleted successfully from SQLite DB.", "success");
            loadContacts();
        } else {
            showToast("Failed to delete contact record.", "error");
        }
    } catch (err) {
        showToast("Error connecting to backend.", "error");
        console.error(err);
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
    updateThemeIcon(nextTheme);
}

function updateThemeIcon(theme) {
    const icon = document.getElementById("theme-icon");
    if (!icon) return;
    icon.className = theme === "light" ? "fa-solid fa-sun" : "fa-solid fa-moon";
}

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    let iconClass = "fa-circle-info";
    if (type === "success") iconClass = "fa-circle-check";
    if (type === "warning") iconClass = "fa-triangle-exclamation";
    if (type === "error")   iconClass = "fa-ban";
    toast.innerHTML = `<i class="fa-solid ${iconClass}"></i><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = "slideOutRight 0.3s forwards";
        toast.addEventListener("animationend", () => toast.remove());
    }, 3500);
}