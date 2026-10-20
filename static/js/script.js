document.addEventListener("DOMContentLoaded", function () {

    const analyzeBtn = document.getElementById("analyzeBtn");

    analyzeBtn.addEventListener("click", async function () {

        console.log("Analyze clicked");

        try {

            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    type: "sms",
                    text: "Congratulations! You won a prize. Click here to claim."
                })
            });

            const data = await response.json();

            console.log("Flask response:", data);

            alert(JSON.stringify(data));

        } catch (error) {

            console.error("ERROR:", error);

            alert("Error: " + error.message);

        }

    });

});