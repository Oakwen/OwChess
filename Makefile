CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra

CEREBRUM_SRC = cerebrum/3.\ inference/cerebrum.c
TEST_DIR = cerebrum/tests

TEST_C = $(TEST_DIR)/test_cerebrum.c
UNIT_C = $(TEST_DIR)/unit_tests.c
CONVERT_C = $(TEST_DIR)/run_nn_convert.c

BIN_TEST = $(TEST_DIR)/test_cerebrum.exe
BIN_UNIT = $(TEST_DIR)/unit_tests.exe
BIN_CONVERT = $(TEST_DIR)/run_nn_convert.exe

.PHONY: all test unit convert clean

all: test unit

test: $(BIN_TEST)
	@echo "Running integration test..."
	@$(BIN_TEST)

unit: $(BIN_UNIT)
	@echo "Running unit tests..."
	@$(BIN_UNIT)

convert: $(BIN_CONVERT)
	@echo "Built run_nn_convert.exe (to convert network.txt to NN_FILE)"

$(BIN_TEST): $(CEREBRUM_SRC) $(TEST_C)
	$(CC) $(CFLAGS) -o "$@" $(CEREBRUM_SRC) $(TEST_C)

$(BIN_UNIT): $(CEREBRUM_SRC) $(UNIT_C)
	$(CC) $(CFLAGS) -o "$@" $(CEREBRUM_SRC) $(UNIT_C)

$(BIN_CONVERT): $(CEREBRUM_SRC) $(CONVERT_C)
	$(CC) $(CFLAGS) -o "$@" $(CEREBRUM_SRC) $(CONVERT_C)

clean:
	-@rm -f $(BIN_TEST) $(BIN_UNIT) $(BIN_CONVERT)
