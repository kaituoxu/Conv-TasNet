CC = gcc

all: sph2pipe create_mix

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
	cd sph2pipe_v2.5/; \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5: sph2pipe_v2.5.tar.gz
	tar xzf sph2pipe_v2.5.tar.gz

create_mix:
	unzip create-speaker-mixtures.zip -d create-speaker-mixtures

clean:
	rm -rf sph2pipe_v2.5/
	rm -rf create-speaker-mixtures/
