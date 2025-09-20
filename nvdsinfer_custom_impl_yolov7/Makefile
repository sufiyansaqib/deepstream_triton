CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=7.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= nvdsparsebbox_yolov7.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -fPIC -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \
	 -DNVDS_VERSION_MAJOR=7 -DNVDS_VERSION_MINOR=1

CFLAGS+= `pkg-config --cflags $(PKGS)`

CFLAGS+= -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
	 -std=c++14 -O2

LIBS:= `pkg-config --libs $(PKGS)`
LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcublas -lstdc++
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm
LIBS+= -Wl,-rpath,$(LIB_INSTALL_DIR)

TARGET:= libnvdsinfer_custom_impl_Yolo.so

all: $(TARGET)

%.o: %.cpp $(INCS) Makefile
	@echo " Compiling: $<"
	$(CXX) -c -o $@ $(CFLAGS) $<

$(TARGET): $(OBJS) Makefile
	@echo " Linking: $@"
	$(CXX) -shared -o $@ $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(TARGET)

install: $(TARGET)
	cp -rv $(TARGET) $(LIB_INSTALL_DIR)