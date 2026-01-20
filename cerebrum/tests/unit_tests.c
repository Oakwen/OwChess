
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../3. inference/cerebrum.h"

static int fail_count = 0;

static void expect(int condition, const char *msg) {
  if (!condition) {
    printf("FAIL: %s\n", msg);
    fail_count++;
  } else {
    printf("ok: %s\n", msg);
  }
}

int test_nn_load(void) {
  int r = nn_load(NN_FILE);
  expect(r == 0, "nn_load should succeed and return 0");
  return r;
}

int test_accumulator_and_add_move(void) {
  NN_Accumulator acc1, acc2;
  nn_init_accumulator(acc1);
  memcpy(acc2, acc1, sizeof(acc1));

  uint64_t whites[6] = {0};
  uint64_t blacks[6] = {0};

  // place white pawn on e2 (square 12)
  whites[0] = 1ULL << 12;
  nn_update_all_pieces(acc1, whites, blacks);

  int eval_with_pawn = nn_evaluate(acc1, 0);
  int eval_baseline = nn_evaluate(acc2, 0);

  expect(eval_with_pawn != eval_baseline,
         "evaluation should change after adding a pawn");

  // move pawn e2->e3 (12->20) using nn_mov_piece
  memcpy(acc2, acc1, sizeof(acc1));
  nn_mov_piece(acc2, 0, 0, 12, 20);
  int eval_after_move = nn_evaluate(acc2, 0);
  expect(eval_after_move != eval_with_pawn,
         "evaluation should change after moving the pawn");

  // delete pawn from e3 and add a pawn for black on e7
  memcpy(acc2, acc1, sizeof(acc1));
  blacks[0] = 1ULL << 52; // e7
  nn_update_all_pieces(acc2, whites, blacks);
  int eval_with_black = nn_evaluate(acc2, 0);
  expect(eval_with_black != eval_with_pawn,
         "evaluation should change after adding opposing pawn");

  return 0;
}

int test_add_del_symmetry(void) {
  NN_Accumulator accA, accB;
  nn_init_accumulator(accA);
  nn_init_accumulator(accB);

  // add pawn then delete, should restore to initial accumulator
  nn_add_piece(accA, 0, 0, 12); // white pawn e2
  nn_del_piece(accA, 0, 0, 12);

  int equal = (memcmp(accA, accB, sizeof(accA)) == 0);
  expect(equal, "adding then deleting a piece should restore accumulator");

  return 0;
}

int main(void) {
  printf("Running cerebrum unit tests...\n");

  int r = test_nn_load();
  (void)r;

  test_accumulator_and_add_move();
  test_add_del_symmetry();

  if (fail_count == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  } else {
    printf("%d TESTS FAILED\n", fail_count);
    return 1;
  }
}
