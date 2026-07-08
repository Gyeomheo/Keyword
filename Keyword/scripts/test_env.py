import sys
with open("test_env.txt", "w") as f:
    f.write(f"Python OK: {sys.version}")
print("TEST_ENV_OK")
