import requests
import time

API_URL = "https://rl-mlops-lab-1.onrender.com/predict"


NUM_REQUESTS = 20

latencies = []
failed_requests = 0

session = requests.Session()

payload = {
    "cart_position": 0.0,
    "cart_velocity": 0.0,
    "pole_angle": 0.03,
    "pole_angular_velocity": 0.0
}

for request_number in range(NUM_REQUESTS):

    start_time = time.time()

    try:
        response = session.post(
            API_URL,
            json=payload,
            timeout=30
        )

        latency = time.time() - start_time

        if response.status_code == 200:
            latencies.append(latency)

            print(
                "Request:",
                request_number + 1,
                "Latency:",
                round(latency, 3),
                "seconds"
            )

        else:
            failed_requests += 1

            print(
                "Request failed:",
                response.status_code,
                response.text
            )

    except requests.RequestException as error:
        failed_requests += 1

        print(
            "Request error:",
            error
        )

print()
print("API Monitoring")
print("--------------")

if latencies:
    print(
        "Average latency:",
        round(
            sum(latencies) / len(latencies),
            3
        ),
        "seconds"
    )

    print(
        "Fastest request:",
        round(
            min(latencies),
            3
        ),
        "seconds"
    )

    print(
        "Slowest request:",
        round(
            max(latencies),
            3
        ),
        "seconds"
    )

else:
    print(
        "No successful requests were recorded."
    )

print(
    "Failed requests:",
    failed_requests
)