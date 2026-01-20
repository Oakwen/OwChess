#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../3. inference/cerebrum.h"

int main(void) {
  int res = nn_load(NN_FILE);
  if (res == 0) {
    printf("nn_load: success\n");
  } else {
    printf("nn_load: failed (%d) -- continuing with zeroed network\n", res);
  }

  NN_Accumulator acc;

  // initialised from biases
  nn_init_accumulator(acc);

  uint64_t whites[6] = {0};
  uint64_t blacks[6] = {0};

  // place white pawn on e2 (a1=0 -> e2 = 8 + 4 = 12)
  whites[0] = 1ULL << 12;
  nn_update_all_pieces(acc, whites, blacks);
  int eval1 = nn_evaluate(acc, 0);
  printf("Eval with white pawn on e2 (white to move): %d\n", eval1);

  // move pawn e2 -> e3 (12 -> 20)
  whites[0] = 1ULL << 20;
  nn_update_all_pieces(acc, whites, blacks);
  int eval2 = nn_evaluate(acc, 0);
  printf("Eval after move e2->e3: %d\n", eval2);

  // add black pawn on e7 (e7 = 48 + 4 = 52)
  blacks[0] = 1ULL << 52;
  nn_update_all_pieces(acc, whites, blacks);
  int eval3 = nn_evaluate(acc, 0);
  printf("Eval after adding black pawn on e7: %d\n", eval3);

  return 0;
}
