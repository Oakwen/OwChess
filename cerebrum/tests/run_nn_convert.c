#include <stdio.h>
#include "../3. inference/cerebrum.h"
#include <sys/stat.h>

int main(void) {
  int r = nn_convert();
  printf("nn_convert returned: %d\n", r);

  struct stat st;
  if (stat(NN_FILE, &st) == 0) {
    printf("Generated %s : %lld bytes\n", NN_FILE, (long long)st.st_size);
  } else {
    printf("Did not find generated file %s\n", NN_FILE);
  }

  return r;
}
