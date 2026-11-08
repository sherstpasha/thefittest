(function () {
    var script = document.querySelector("script[data-current-version][data-version-map]");
    if (!script) {
        return;
    }

    var currentVersion = script.dataset.currentVersion;
    var versions = JSON.parse(script.dataset.versionMap);
    var current = versions[currentVersion];
    if (!current) {
        return;
    }

    var currentRoot = new URL("../", script.src);
    var siteRoot = new URL("../".repeat(current.path.split("/").length), currentRoot);
    var page = decodeURI(window.location.pathname.slice(currentRoot.pathname.length));
    page = page || "index.html";

    function targetFor(version) {
        var target = versions[version];
        var targetPage = target.pages.indexOf(page) !== -1 ? page : "index.html";
        var url = new URL(target.path + "/" + targetPage, siteRoot);
        url.search = window.location.search;
        url.hash = version === currentVersion ? window.location.hash : "";
        return url.href;
    }

    function buildSwitcher() {
        var menu = document.querySelector(".wy-menu.wy-menu-vertical");
        if (!menu || menu.querySelector(".version-nav-section")) {
            return;
        }

        var caption = document.createElement("p");
        caption.className = "caption version-nav-section";
        caption.setAttribute("role", "heading");
        var title = document.createElement("span");
        title.className = "caption-text";
        title.textContent = "Versions:";
        caption.appendChild(title);

        var list = document.createElement("ul");
        list.className = "version-nav-list";
        Object.keys(versions).forEach(function (version) {
            var item = document.createElement("li");
            item.className = "toctree-l1";
            var link = document.createElement("a");
            link.className = "reference internal";
            link.href = targetFor(version);
            link.textContent = version + (version === currentVersion ? " (current)" : "");
            if (version === currentVersion) {
                link.setAttribute("aria-current", "page");
            }
            item.appendChild(link);
            list.appendChild(item);
        });
        menu.appendChild(caption);
        menu.appendChild(list);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", buildSwitcher);
    } else {
        buildSwitcher();
    }
})();
