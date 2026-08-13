document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("analyzeForm");
    const loadingContainer = document.getElementById("loadingContainer");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const progressBar = document.getElementById("progressBar");
    const elapsedTime = document.getElementById("elapsedTime");

    if (form) {
        form.addEventListener("submit", () => {
            if (loadingContainer) {
                loadingContainer.classList.remove("hidden");
            }
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = "Analyzing...";
            }

            let progress = 0;
            const start = Date.now();
            const progressTimer = setInterval(() => {
                progress = Math.min(progress + 3, 95);
                if (progressBar) {
                    progressBar.style.width = `${progress}%`;
                }
            }, 1000);

            const elapsedTimer = setInterval(() => {
                const seconds = Math.floor((Date.now() - start) / 1000);
                if (elapsedTime) {
                    elapsedTime.textContent = `Elapsed: ${seconds}s`;
                }
            }, 1000);

            window.addEventListener("beforeunload", () => {
                clearInterval(progressTimer);
                clearInterval(elapsedTimer);
            });
        });
    }

    const setupThemeMenu = () => {
        const html = document.documentElement;
        const menuTrigger = document.getElementById("menuTrigger");
        const menuDropdown = document.getElementById("menuDropdown");
        const printBtn = document.getElementById("printBtn");

        const applyTheme = (theme) => {
            const selected = theme || "system";
            if (selected === "system") {
                const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                html.setAttribute("data-theme", prefersDark ? "dark" : "light");
            } else {
                html.setAttribute("data-theme", selected);
            }
        };

        const savedTheme = localStorage.getItem("themePreference") || "system";
        applyTheme(savedTheme);

        if (menuTrigger && menuDropdown) {
            menuTrigger.addEventListener("click", () => {
                const expanded = menuTrigger.getAttribute("aria-expanded") === "true";
                menuTrigger.setAttribute("aria-expanded", String(!expanded));
                menuDropdown.classList.toggle("hidden");
            });

            document.addEventListener("click", (event) => {
                if (!menuTrigger.contains(event.target) && !menuDropdown.contains(event.target)) {
                    menuTrigger.setAttribute("aria-expanded", "false");
                    menuDropdown.classList.add("hidden");
                }
            });
        }

        document.querySelectorAll("[data-theme]").forEach((button) => {
            button.addEventListener("click", () => {
                const value = button.getAttribute("data-theme");
                localStorage.setItem("themePreference", value || "system");
                applyTheme(value || "system");
            });
        });

        if (printBtn) {
            printBtn.addEventListener("click", () => {
                window.print();
            });
        }
    };
    setupThemeMenu();

    const setupSentimentPieChart = () => {
        const chartDataEl = document.getElementById("sentimentChartData");
        const canvas = document.getElementById("sentimentPieChart");
        if (!chartDataEl || !canvas || typeof Chart === "undefined") {
            return;
        }

        const values = [
            Number.parseInt(chartDataEl.dataset.positive || "0", 10),
            Number.parseInt(chartDataEl.dataset.negative || "0", 10),
            Number.parseInt(chartDataEl.dataset.neutral || "0", 10),
        ];
        const total = values.reduce((acc, current) => acc + current, 0);

        const pieLabelPlugin = {
            id: "pieLabelPlugin",
            afterDatasetsDraw(chart) {
                const { ctx } = chart;
                const meta = chart.getDatasetMeta(0);
                ctx.save();
                ctx.fillStyle = "#111111";
                ctx.font = "600 16px Inter";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                meta.data.forEach((arc, index) => {
                    const value = values[index];
                    if (value <= 0 || total === 0) {
                        return;
                    }
                    const percent = `${((value / total) * 100).toFixed(1)}%`;
                    const { x, y } = arc.tooltipPosition();
                    ctx.fillText(percent, x, y);
                });
                ctx.restore();
            },
        };

        // Slight explode effect using array offset values.
        new Chart(canvas.getContext("2d"), {
            type: "pie",
            data: {
                labels: ["Positive", "Negative", "Neutral"],
                datasets: [
                    {
                        data: values,
                        backgroundColor: ["#4CAF50", "#F44336", "#FFC107"],
                        borderColor: "#f0f0f0",
                        borderWidth: 2,
                        offset: [12, 10, 8],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: getComputedStyle(document.body).getPropertyValue("--text").trim() || "#111111",
                            font: { size: 16, family: "Inter" },
                            boxWidth: 20,
                        },
                    },
                    title: {
                        display: true,
                        text: "Sentiment Distribution",
                        color: getComputedStyle(document.body).getPropertyValue("--text").trim() || "#111111",
                        font: { size: 22, weight: "600", family: "Inter" },
                    },
                },
            },
            plugins: [pieLabelPlugin],
        });
    };
    setupSentimentPieChart();

    document.querySelectorAll(".accordion-toggle").forEach((button) => {
        button.addEventListener("click", () => {
            const content = button.nextElementSibling;
            if (content) {
                content.classList.toggle("open");
            }
        });
    });

    document.querySelectorAll(".copy-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            const text = button.getAttribute("data-copy") || "";
            try {
                await navigator.clipboard.writeText(text);
                button.textContent = "Copied";
                setTimeout(() => {
                    button.textContent = "Copy";
                }, 1200);
            } catch (error) {
                button.textContent = "Failed";
            }
        });
    });

    const resultTop = document.getElementById("resultsTop");
    if (resultTop) {
        resultTop.scrollIntoView({ behavior: "smooth", block: "start" });
    }
});
