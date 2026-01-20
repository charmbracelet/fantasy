![kronk logo](https://github.com/ardanlabs/kronk/blob/main/images/project/kronk_banner.jpg?raw=true&v5)

Copyright 2025 Ardan Labs  
hello@ardanlabs.com

# Kronk

https://github.com/ardanlabs/kronk

This project lets you use Go for hardware accelerated local inference with llama.cpp directly integrated into your applications via the [yzma](https://github.com/hybridgroup/yzma) module. Kronk provides a high-level API that feels similar to using an OpenAI compatible API.

This project also provides a model server for chat completions, responses, embeddings, and reranking. The server is compatible with the OpebWebUI and Cline projects.

Here is the current [catalog](https://github.com/ardanlabs/kronk_catalogs) of models that have been verified to work with Kronk.

To see all the documentation, clone the project and run the Kronk Model Server:

```shell
$ make kronk-server

$ make website
```

You can also install Kronk, run the Kronk Model Server, and open the browser to localhost:8080

```shell
$ go install github.com/ardanlabs/kronk/cmd/kronk@latest

$ kronk server start
```

## Owner Information

```
Name:    Bill Kennedy
Company: Ardan Labs
Title:   Managing Partner
Email:   bill@ardanlabs.com
Twitter: goinggodotnet
```

## Install Kronk

To install the Kronk tool run the following command:

```shell
$ go install github.com/ardanlabs/kronk/cmd/kronk@latest

$ kronk --help
```

## Architecture

The architecture of Kronk is designed to be simple and scalable.

Watch this [video](https://www.youtube.com/live/gjSrYkYc-yo) to learn more about the project and the architecture.

### SDK

The Kronk SDK allows you to write applications that can diectly interact with local open source GGUF models (supported by llama.cpp) that provide inference for text and media (vision and audio).

![api arch](https://github.com/ardanlabs/kronk/blob/main/images/design/sdk.png?raw=true&v1)

Check out the [examples](#examples) section below.

### Kronk Model Server (KMS)

If you want an OpenAI compatible model server, the Kronk model server leverages the power of the Kronk API to give you a concurrent and scalable web api.

Run `make kronk-server` to check it out.

## Models

Kronk uses models in the GGUF format supported by llama.cpp. You can find many models in GGUF format on Hugging Face (over 147k at last count):

https://huggingface.co/models?library=gguf&sort=trending

## Support

Kronk currently has support for over 94% of llama.cpp functionality thanks to yzma. See the yzma [ROADMAP.md](https://github.com/hybridgroup/yzma/blob/main/ROADMAP.md) for the complete list.

You can use multimodal models (image/audio) and text language models with full hardware acceleration on Linux, on macOS, and on Windows.

| OS      | CPU          | GPU                             |
| ------- | ------------ | ------------------------------- |
| Linux   | amd64, arm64 | CUDA, Vulkan, HIP, ROCm, SYCL   |
| macOS   | arm64        | Metal                           |
| Windows | amd64        | CUDA, Vulkan, HIP, SYCL, OpenCL |

Whenever there is a new release of llama.cpp, the tests for yzma are run automatically. Kronk runs tests once a day and will check for updates to llama.cpp. This helps us stay up to date with the latest code and models.
