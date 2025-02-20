#define STORAGE_CLASS register
#define TYPE  float

void pin_down(TYPE * px, int LENGTH)
{
	STORAGE_CLASS int    i;

	for (i = 0; i < LENGTH; ++i) {
		*px++ = 1.0;
	}

}