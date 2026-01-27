$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

$items = @(
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_sun.jpg"; Dest = "src/orrery_app/data/textures/8k/sun.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_mercury.jpg"; Dest = "src/orrery_app/data/textures/8k/mercury.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_venus_surface.jpg"; Dest = "src/orrery_app/data/textures/8k/venus_surface.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg"; Dest = "src/orrery_app/data/textures/8k/earth_day.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_earth_clouds.jpg"; Dest = "src/orrery_app/data/textures/extras/earth_clouds_8k.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_moon.jpg"; Dest = "src/orrery_app/data/textures/8k/moon.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_mars.jpg"; Dest = "src/orrery_app/data/textures/8k/mars.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_jupiter.jpg"; Dest = "src/orrery_app/data/textures/8k/jupiter.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/8k_saturn.jpg"; Dest = "src/orrery_app/data/textures/8k/saturn.jpg" },

    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_sun.jpg"; Dest = "src/orrery_app/data/textures/2k/sun.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_mercury.jpg"; Dest = "src/orrery_app/data/textures/2k/mercury.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_venus_surface.jpg"; Dest = "src/orrery_app/data/textures/2k/venus_surface.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"; Dest = "src/orrery_app/data/textures/2k/earth_day.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_earth_clouds.jpg"; Dest = "src/orrery_app/data/textures/extras/earth_clouds_2k.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_moon.jpg"; Dest = "src/orrery_app/data/textures/2k/moon.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_mars.jpg"; Dest = "src/orrery_app/data/textures/2k/mars.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_jupiter.jpg"; Dest = "src/orrery_app/data/textures/2k/jupiter.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_saturn.jpg"; Dest = "src/orrery_app/data/textures/2k/saturn.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_uranus.jpg"; Dest = "src/orrery_app/data/textures/2k/uranus.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_neptune.jpg"; Dest = "src/orrery_app/data/textures/2k/neptune.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_ceres_fictional.jpg"; Dest = "src/orrery_app/data/textures/2k/ceres.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/4k_ceres_fictional.jpg"; Dest = "src/orrery_app/data/textures/4k/ceres.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_eris_fictional.jpg"; Dest = "src/orrery_app/data/textures/2k/eris.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/4k_eris_fictional.jpg"; Dest = "src/orrery_app/data/textures/4k/eris.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_haumea_fictional.jpg"; Dest = "src/orrery_app/data/textures/2k/haumea.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/4k_haumea_fictional.jpg"; Dest = "src/orrery_app/data/textures/4k/haumea.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/2k_makemake_fictional.jpg"; Dest = "src/orrery_app/data/textures/2k/makemake.jpg" },
    @{ Url = "https://www.solarsystemscope.com/textures/download/4k_makemake_fictional.jpg"; Dest = "src/orrery_app/data/textures/4k/makemake.jpg" },

    @{ Url = "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/january/world.200401.3x21600x10800.jpg"; Dest = "src/orrery_app/data/textures/hires/earth_21600.jpg" },

    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/ura1vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/ariel.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/ura2vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/umbriel.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/ura3vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/titania.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/ura4vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/oberon.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/ura5vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/miranda.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/nep1vuu2.jpg"; Dest = "src/orrery_app/data/textures/low/triton.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/plu0rss1.jpg"; Dest = "src/orrery_app/data/textures/low/pluto.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/plu1rss1.jpg"; Dest = "src/orrery_app/data/textures/low/charon.jpg" },

    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/mar1kuu2.jpg"; Dest = "src/orrery_app/data/textures/low/phobos.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/mar2kuu2.jpg"; Dest = "src/orrery_app/data/textures/low/deimos.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/jup1vss2.jpg"; Dest = "src/orrery_app/data/textures/low/io.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/jup2vss2.jpg"; Dest = "src/orrery_app/data/textures/low/europa.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/jup3vss2.jpg"; Dest = "src/orrery_app/data/textures/low/ganymede.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/jup4vss2.jpg"; Dest = "src/orrery_app/data/textures/low/callisto.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat1vss2.jpg"; Dest = "src/orrery_app/data/textures/low/mimas.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat2vss2.jpg"; Dest = "src/orrery_app/data/textures/low/enceladus.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat3vss2.jpg"; Dest = "src/orrery_app/data/textures/low/tethys.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat4vss2.jpg"; Dest = "src/orrery_app/data/textures/low/dione.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat5vss2.jpg"; Dest = "src/orrery_app/data/textures/low/rhea.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat6fss1.jpg"; Dest = "src/orrery_app/data/textures/low/titan.jpg" },
    @{ Url = "https://maps.jpl.nasa.gov/tmaps/pix/sat8vss2.jpg"; Dest = "src/orrery_app/data/textures/low/iapetus.jpg" }
)

foreach ($item in $items) {
    $dest = Join-Path $repoRoot $item.Dest
    if (Test-Path $dest) {
        continue
    }

    $dir = Split-Path $dest
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    Write-Host "Downloading $($item.Url) -> $($item.Dest)"
    $ok = $true
    try {
        curl.exe -L -f $item.Url -o $dest
    } catch {
        $ok = $false
    }

    if (-not $ok -or -not (Test-Path $dest) -or (Get-Item $dest).Length -le 0) {
        Write-Warning "Failed to download $($item.Url)"
        if (Test-Path $dest) {
            Remove-Item $dest -Force
        }
        continue
    }

    $sizeBytes = (Get-Item $dest).Length
    if ($sizeBytes -gt 500MB) {
        Write-Warning "Texture exceeds 500MB: $($item.Dest) ($([math]::Round($sizeBytes / 1MB, 1)) MB)"
    }
}
