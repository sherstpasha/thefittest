(function () {
    function scriptData() {
        var scripts = document.querySelectorAll("script[data-current-version]");
        return scripts[scripts.length - 1] || {};
    }

    function splitVersions(value) {
        return (value || "")
            .split(",")
            .map(function (version) {
                return version.trim();
            })
            .filter(Boolean);
    }

    function pageForVersion(targetVersion, versions) {
        var path = window.location.pathname;
        var suffix = window.location.search + window.location.hash;

        for (var i = 0; i < versions.length; i += 1) {
            var marker = "/" + versions[i] + "/";
            var markerIndex = path.indexOf(marker);
            if (markerIndex !== -1) {
                return (
                    path.slice(0, markerIndex + 1) +
                    targetVersion +
                    "/" +
                    path.slice(markerIndex + marker.length) +
                    suffix
                );
            }
        }

        return targetVersion + "/index.html" + suffix;
    }

    function buildSwitcher() {
        var data = scriptData();
        var currentVersion = data.dataset ? data.dataset.currentVersion : "";
        var versions = splitVersions(data.dataset ? data.dataset.versions : "");
        var mount = document.querySelector(".wy-side-nav-search");

        if (!mount || versions.length === 0) {
            return;
        }

        var wrapper = document.createElement("div");
        wrapper.className = "version-switcher";

        var label = document.createElement("label");
        label.className = "version-switcher-label";
        label.setAttribute("for", "version-switcher-select");
        label.textContent = "Documentation version";

        var select = document.createElement("select");
        select.id = "version-switcher-select";
        select.className = "version-switcher-select";
        select.setAttribute("aria-label", "Documentation version");

        versions.forEach(function (version) {
            var option = document.createElement("option");
            option.value = version;
            option.textContent = version;
            option.selected = version === currentVersion;
            select.appendChild(option);
        });

        select.addEventListener("change", function () {
            window.location.href = pageForVersion(select.value, versions);
        });

        wrapper.appendChild(label);
        wrapper.appendChild(select);
        mount.appendChild(wrapper);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", buildSwitcher);
    } else {
        buildSwitcher();
    }
})();
