# Kronk

> [!IMPORTANT]
> The Kronk provider for Fantasy is considered experimental!
> The API and behavior can change until we consider it stable.

[Kronk][kronk] is a Go package by [ArdanLabs][ardanlabs] that uses [yzma][yzma]
under the hood to provide hardware accelerated local inference with
[llama.cpp][llama.cpp] directly integrated into your applications.
Kronk provides a high-level API that feels similar to using an OpenAI compatible
API.

When using the Kronk provider in fantasy, you only need to specify the model you
want to use and it'll be automatically downloaded in your machine and used for
inference.

To see which models are available for you, see the [Kronk Catalog][catalog].

Examples on how to use it are available in [`examples/kronk`][examples].

[kronk]: https://github.com/ardanlabs/kronk
[ardanlabs]: https://github.com/ardanlabs
[yzma]: https://github.com/hybridgroup/yzma
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[catalog]: https://github.com/ardanlabs/kronk_catalogs
[examples]: https://github.com/charmbracelet/fantasy/tree/main/examples/kronk
