# A dockerfile used to extract a manylinux2010 compatible sysroot
FROM gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010

ADD fix_symlinks.sh .

RUN /fix_symlinks.sh /dt8 && \
    cd /dt8 && \
    tar -cvzf /sysroot.tar.gz .